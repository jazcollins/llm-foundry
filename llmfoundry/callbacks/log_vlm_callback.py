# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations from a set of prompts and images for a VLM."""

import logging
import time
from typing import Any, List, Optional, Union, cast

from composer.core import Callback, Event, State, Time, get_precision_context
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import create_interval_scheduler, dist
from composer.utils.import_helpers import MissingConditionalImportError
import wandb

from llmfoundry.data.finetuning.tasks import _process_image
from PIL import Image
import requests
import torch

log = logging.getLogger(__name__)

# TODO should grab this from elsewhere in case we want to change it?
SYSTEM = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'

class GenerateVLM(Callback):
    """Periodically log generations from a set of prompts and images for a VLM.

    Args:
        prompts (List[str]): The list of prompts you would like to produce generations for
        image_urls (List[str]): Corresponding image urls for prompt list
        interval (Union[str, int, :class:`.Time`]): The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        batch_size (Optional[int]): Size of a prompt batch for generation. If None, defaults to the number of prompts.
        kwargs: All kwargs will be passed along to the call to generate. This is for things like `do_sample`, `top_p`, etc
    """

    def __init__(self,
                 prompts: List[str],
                 image_urls: List[str],
                 interval: Union[str, int, Time],
                 batch_size: Optional[int] = None,
                 **kwargs: Any):
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp',
                                                conda_package='transformers',
                                                conda_channel='conda-forge') from e
        del transformers
        self.prompts = prompts
        self.image_urls = image_urls
        self.generate_kwargs = kwargs
        self.batch_size = batch_size if batch_size is not None else len(prompts)
        self.check_interval = create_interval_scheduler(interval, include_end_of_training=True)
        self.last_generate_batch: Optional[Time] = None

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if state.get_elapsed_duration() is not None and self.check_interval(
                state, event) and self.last_generate_batch != state.timestamp.batch:
            start = time.time()
            self.generate(state, logger)
            diff = time.time() - start
            log.info(f'Generate callback ran in {diff} seconds for {len(self.prompts)} prompts')

    def generate(self, state: State, logger: Logger):
        self.last_generate_batch = state.timestamp.batch

        model = state.model.module if state.is_model_ddp else state.model
        if not isinstance(model, HuggingFaceModel):  # TODO: Extend to support any models that have a generate method.
            raise ValueError(f'Expected HuggingFaceModel, but got {model.__class__.__name__}')

        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            raise ValueError(
                f'Model {model.__class__.__name__} does not have a tokenizer which is required for generation.')
        tokenizer = model.tokenizer

        from transformers import PreTrainedTokenizerBase
        tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

        # Set to evaluation mode and stash the original mode.
        original_mode = model.training
        model.eval()
        device = state.device

        aug_prompts = []
        images = []
        orig_images = []
        for prompt, url in zip(self.prompts, self.image_urls):
            if tokenizer.chat_template is not None:
                formatted_convo = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '<image>\n'+prompt}]
                aug_prompt = tokenizer.apply_chat_template(formatted_convo,
                                                           tokenize=False,
                                                           add_generation_prompt=True)
            else:
                aug_prompt = '<image>\n'+prompt
            aug_prompts.append(aug_prompt)

            image = Image.open(requests.get(url, stream=True).raw)
            image = image.convert("RGBA")
            orig_images.append(image)
            processed_img = _process_image(image)
            images.append(processed_img)

        # Stash the original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(aug_prompts, return_tensors='pt', padding=True)

        all_input_ids = tokenized_input['input_ids']
        all_attn_masks = tokenized_input['attention_mask']
        all_images = torch.stack(images)

        output_token_ids = []
        # dummy forward call needed for FSDP to work consistently
        model.dummy_forward_called = False

        n_prompts = len(self.prompts)
        for start in range(0, n_prompts, self.batch_size):
            end = min(start + self.batch_size, n_prompts)
            input_ids = all_input_ids[start:end]  # pyright: ignore[reportGeneralTypeIssues]
            attn_mask = all_attn_masks[start:end]  # pyright: ignore[reportGeneralTypeIssues]
            img = all_images[start:end] # pyright: ignore[reportGeneralTypeIssues]

            # Move batch to device.
            input_ids = device.tensor_to_device(input_ids)
            attn_mask = device.tensor_to_device(attn_mask)
            img = device.tensor_to_device(img)
            
            with get_precision_context(state.precision):
                output_token_ids.extend(
                    model.generate(  # type: ignore
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        pixel_values=img,
                        synced_gpus=dist.get_world_size() > 1,
                        **self.generate_kwargs,
                    ))

        if dist.get_global_rank() == 0:
            # Process prompts and outputs into a table.
            rows = []
            input_tokens_len = all_input_ids.shape[1]  # pyright: ignore[reportGeneralTypeIssues]
            for i, prompt in enumerate(self.prompts):
                image = orig_images[i]
                aug_prompt = aug_prompts[i]
                output_tokens = output_token_ids[i][input_tokens_len:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

                rows.append([wandb.Image(image), prompt, aug_prompt, output_text])

            # There are problems with using the MosaicMLLogger and/or console_logger --> the images can't get decoded correctly 
            # Hacky but just log to the WandB logger
            table = wandb.Table(columns=['img', 'prompt', 'aug_prompt', 'generation'], rows=rows)
            wandb.log({'generations': table}, state.timestamp.batch.value)

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)