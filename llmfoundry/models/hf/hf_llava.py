
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face LLaVA model wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Mapping
import torch
from torch import nn

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.utils import dist
from omegaconf import DictConfig
from transformers import (AutoConfig, PreTrainedTokenizerBase, CLIPVisionConfig,
                          LlavaForConditionalGeneration, LlavaConfig, AutoModelForCausalLM,
                          AutoModel)
from composer.models import HuggingFaceModel
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.utils import (adapt_tokenizer_for_denoising,
                                     init_empty_weights)

from collections import UserDict
from transformers.utils.generic import ModelOutput
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaMultiModalProjector
from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp

__all__ = ['ComposerHFLLaVa']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class ComposerHFLLaVa(HuggingFaceModel):
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

        # # Get the device we want to initialize, and use the
        # # resolved version to initialize the HF model
        # resolved_init_device = hf_get_init_device(init_device)

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            om_model_config.pretrained = False

        train_metrics = [LanguageCrossEntropy()]
        eval_metrics = [LanguageCrossEntropy()]

        if om_model_config.pretrained:
            model = LlavaForConditionalGeneration.from_pretrained(
                om_model_config.pretrained_model_name_or_path,
                config=config,
                cache_dir="/tmp/model_cache/")           
        else:
            # bakklava style
            vision_tower = 'openai/clip-vit-large-patch14-336'
            vision_config = CLIPVisionConfig.from_pretrained(vision_tower)

            model_name_or_path = 'mistralai/Mistral-7B-v0.1'
            text_config = AutoConfig.from_pretrained(model_name_or_path)

            llava_config = LlavaConfig(vision_config, text_config)
            model = LlavaForConditionalGenerationForTraining(model_name_or_path, vision_tower, llava_config)
            # TODO i assume this model is fully scratch in shape of bakklava
            # want to rewrite LlavaForConditionalGeneration init function to load from pretrained?

        # temp fix
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        # see https://github.com/huggingface/transformers/issues/26969
        #   File "/usr/lib/python3/dist-packages/transformers/models/mistral/modeling_mistral.py", line 293, in forward
        # raise ValueError(
        # ValueError: Attention mask should be of size (1, 1, 1599, 3198), but is torch.Size([1, 1, 1599, 1599] 

        # TODO want to set differently for different phases
        model.vision_tower.requires_grad_(False)
        model.language_model.requires_grad_(False)
        # model.multi_modal_projector.requires_grad_(True)        

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())

        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].'
            )
        
        prepare_hf_model_for_fsdp(self.model, init_device)

        # This provides support for meta initialization when using FSDP
        self.model.param_init_fn = lambda module: self.model._init_weights(
            module)
        
    def get_targets(self, batch: Mapping) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

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

    def loss(self, outputs: LlavaCausalLMOutputWithPast,
             batch: Mapping) -> torch.Tensor:
        targets = self.get_targets(batch)
        return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)),
                            targets.view(-1))
    

class LlavaForConditionalGenerationForTraining(LlavaForConditionalGeneration):
    '''
        Overwrite LlavaForConditionalGeneration's init function so we can load CLIP and LLM weights.
    '''
    def __init__(self, text_model_name, vision_model_name, config: LlavaConfig):
        # TODO can probably infer text_model_name and vision_model_name from config..
        super().__init__(config)
        self.vision_tower = AutoModel.from_pretrained(vision_model_name)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()