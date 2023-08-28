import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import math
import numpy as np
import tqdm
from itertools import chain
from typing import Dict, List
from dataclasses import dataclass
import os
import time

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin
import datasets
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers import BatchEncoding
from deepspeed.ops.adam import FusedAdam

import ray
from ray import train
from ray.air import session, Checkpoint
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.huggingface import AccelerateTrainer
import ray.train as train
from ray.train.torch import TorchTrainer
torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )


        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch


    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length




def load_data():
    # get c4
    training_data = datasets.load_dataset('c4', 'en', streaming=True,) \
                            .remove_columns(['timestamp', 'url'])
    return training_data


def train_func(config: dict):
    batch_size = config["batch_size"]
    epochs = config["epochs"]


    DEEPSPEED_CONFIG_ZERO_TWO = {
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
    }


    # uses more GPU RAM than V2, slightly slower; ~ 63s a step
    DEEPSPEED_CONFIG_ZERO_THREE = {
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
    }


    # ~60s a step
    DEEPSPEED_CONFIG_ZERO_THREE_V2 = {
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": True
}

    # attach ds config
    ds_plugin = DeepSpeedPlugin(hf_ds_config=DEEPSPEED_CONFIG_ZERO_THREE_V2)
    ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = batch_size

    
    # tokenize fn
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_attention_mask=False)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result


    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-xxl', use_fast=True)  # 11 B / google/t5-v1_1-base

    # init config
    config = AutoConfig.from_pretrained('google/t5-v1_1-xxl', vocab_size=len(tokenizer))
    config.dropout_rate = 0.0
    config.use_cache = False

    # init model
    model = T5ForConditionalGeneration(config=config)

    # get lengths
    noise_density = 0.15
    input_length = 512
    mean_noise_span_length = 3.0

    # set lengths
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=input_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )

    # init data collator; will take care of randomly masking the token
    data_collator = DataCollatorForT5MLM(tokenizer=tokenizer,
                                        noise_density=noise_density,
                                        mean_noise_span_length=mean_noise_span_length,
                                        input_length=input_length,
                                        target_length=targets_length,
                                        pad_token_id=config.pad_token_id,
                                        decoder_start_token_id=config.decoder_start_token_id,
    )



    # Initialize the Accelerator
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=1, adjust_scheduler=True)
    accelerator = Accelerator(mixed_precision='bf16',  # 'no' for tf32
                                gradient_accumulation_plugin=gradient_accumulation_plugin,
                                device_placement=True,
                                deepspeed_plugin=ds_plugin,
                                )

    
    # get data
    training_data = load_data()

    # first map
    tokenized_datasets = training_data.map(
        tokenize_function,
        batched=True,
        remove_columns='text',
    )

    # second map
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    # init loader
    train_dataloader = DataLoader(tokenized_datasets['train'],
                                  prefetch_factor=4,
                                  batch_size=batch_size,
                                  collate_fn=data_collator,
                                  num_workers=8)

    
    # optim
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=0.01)

    # scheduler
    lr_scheduler = LambdaLR(optimizer, lambda step: min(1e-2, 1.0 / math.sqrt(step)) / 0.01 if step else 1e-2 / 0.01)

    # prepare
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

    # params
    target_token_train = 4300000000
    tokens_per_step = 512 * batch_size * 8  # num_workers
    max_train_steps = target_token_train // tokens_per_step # 65536  # T5 1.1 paper

    if accelerator.is_main_process:
        print(f"Starting training ... max train steps: {max_train_steps}")

    # train loop
    for epoch in range(1, epochs + 1):
        print(f" Starting epoch {epoch}")
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)

        for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=max_train_steps):

            with accelerator.accumulate(model):

                s_fwd = time.time()
                outputs = model(**batch)

                # loss / store
                loss = outputs.loss
                loss_sum += loss

                # backward
                accelerator.backward(loss)

                # update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                e_opt_step = time.time()
            
            if accelerator.is_main_process:
                accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()} step-time: {e_opt_step - s_fwd}"
                )
            
            # end time
            e_epoch = time.time()

            # end at first stage
            if step >= max_train_steps:
                break

        aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()
        session.report(
            {
                "epoch": epoch,
                "iteration": step,
                "train_loss_batch": aggregated_loss,
                "avg_train_loss_epoch": loss_sum.item() / (step + 1),
                "num_iterations": step + 1,
                "train_time_per_epoch": e_epoch - s_epoch,
                "learning_rate": lr_scheduler.get_lr()[0],
            }
        )
  

    
  



if __name__ == '__main__':

 
    # google/t5-v1_1-xxl; 13B model at 2 batch_size ~ 78 gb memory per A100
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

    trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 1, "batch_size": 2},  # 96 batch_size ~ 78 gb of memory use per A100 @ bf16; tf32 much slower / lower batch size
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    run_config=run_config,
)

    result = trainer.fit()
