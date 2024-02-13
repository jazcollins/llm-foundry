
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face LLaVA model wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Mapping
import torch

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.utils import dist
from omegaconf import DictConfig
from transformers import (AutoConfig, PreTrainedTokenizerBase,
                          LlavaForConditionalGeneration, LlavaConfig,
                          CLIPVisionConfig, MistralConfig, LlamaConfig)

from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.utils import (adapt_tokenizer_for_denoising,
                                     init_empty_weights)

from collections import UserDict
from transformers.utils.generic import ModelOutput

__all__ = ['ComposerHFLLaVa']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class ComposerHFLLaVa(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a T5.

    Note: This function uses `transformers.LlavaForConditionalGeneration`. Future releases
        will expand support to more general classes of HF Encoder-Decoder models.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF model (e.g., `llava-hf/bakLlava-v1-hf` to instantiate a bakllava base config).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path. Default: ``{}``.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
            cfg.z_loss (float, optional): The coefficient of the z-loss. If >0.0, this
                the z-loss will be multiplied by this value before being added to the
                standard loss term. Default: ``0.0``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self, om_model_config: DictConfig,
                 tokenizer: PreTrainedTokenizerBase):
        config = AutoConfig.from_pretrained(
            om_model_config.pretrained_model_name_or_path,
            trust_remote_code=om_model_config.get('trust_remote_code', True),
            use_auth_token=om_model_config.get('use_auth_token', False),
        )

        # set config overrides
        for k, v in om_model_config.get('config_overrides', {}).items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f'Config dict override got unknown keys. ' +
                        f'Extra keys: {extra_keys}. ' +
                        f'Expected (a subset of) keys: {list(attr.keys())}.')
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        init_device = om_model_config.get('init_device', 'cpu')

        # Get the device we want to initialize, and use the
        # resolved version to initialize the HF model
        resolved_init_device = hf_get_init_device(init_device)

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            om_model_config.pretrained = False

        # Initializing a CLIP-vision config
        vision_config = CLIPVisionConfig()

        # Initializing a Mistral config
        text_config = MistralConfig()

        llava_config = LlavaConfig(vision_config, text_config)

        if resolved_init_device == 'cpu':
            if om_model_config.pretrained:
                model = LlavaForConditionalGeneration.from_pretrained(
                    om_model_config.pretrained_model_name_or_path,
                    config=config,
                    cache_dir="/tmp/model_cache/")
                model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
                # see https://github.com/huggingface/transformers/issues/26969
                #   File "/usr/lib/python3/dist-packages/transformers/models/mistral/modeling_mistral.py", line 293, in forward
                # raise ValueError(
                # ValueError: Attention mask should be of size (1, 1, 1599, 3198), but is torch.Size([1, 1, 1599, 1599])

                # TODO want to set differently for different phases
                model.vision_tower.requires_grad_(False)
                model.language_model.requires_grad_(False)
                # model.multi_modal_projector.requires_grad_(True)
                
            else:
                # TODO
                model = LlavaForConditionalGeneration(llava_config)
        elif resolved_init_device == 'meta':
            if om_model_config.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                # TODO
                model = LlavaForConditionalGeneration(llava_config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        metrics = [
            LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX),
            MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
        ]

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          z_loss=om_model_config.get(
                                              'z_loss', 0.0),
                                          init_device=init_device)

        return composer_model
    
    def forward(self, batch: Mapping):
        max_seq_len = batch['input_ids'].shape[1]

        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {
                k: v for k, v in batch.items() if k in self.model_forward_args
            }
            output = self.model(**batch)  # type: ignore (thirdparty)

            # TODO not a great fix
            output.logits = output.logits[:,-max_seq_len:]

        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output