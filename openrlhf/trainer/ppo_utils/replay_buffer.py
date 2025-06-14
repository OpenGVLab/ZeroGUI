import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


from .experience_maker import Experience
from .data_processor import BaseDataProcessor


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    ref_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    visual_inputs: Optional[dict]
    # pixel_values: Optional[torch.Tensor] = None
    # image_flags: Optional[torch.Tensor] = None



def split_experience_batch(experience: Experience, data_processor: Optional[BaseDataProcessor]) -> List[BufferItem]:
    """Splits a batch of experience data into individual experience items.

    This function takes a batched `Experience` object and splits it into a list of `BufferItem`
    instances, each representing a single experience in the batch. Tensors in the input (e.g.,
    `sequences`, `action_log_probs`, `values`, etc.) are split along the first dimension (batch dimension).
    The `info` dictionary values are also split, with tensor values converted to Python scalars.

    Args:
        experience (Experience): 
            A batched experience object containing tensors and metadata. The batch size is inferred from
            the length of `experience.sequences`.

    Returns:
        List[BufferItem]: 
            A list of `BufferItem` objects, each containing non-batched tensors and metadata for a single
            experience instance.

    Note:
        All tensors in the `info` dictionary of `experience` must be scalar tensors (i.e., with exactly one element).
        Non-scalar tensors in `info` will trigger an assertion error.
    """
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "ref_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",

    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v) if isinstance(v, torch.Tensor) else v
        assert batch_size == len(vals), f"batch_size: {batch_size}, len(vals): {len(vals)}, key: {k}, v: {v}"
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv
            
    if data_processor is not None:
        visual_inputs_batch = experience.visual_inputs
        visual_inputs_batch['input_ids'] = experience.sequences
        visual_inputs_chunks = data_processor.split_input_batch(visual_inputs_batch)
        for i, visual_inputs in enumerate(visual_inputs_chunks):
            visual_inputs.pop('input_ids')
            batch_kwargs[i]["visual_inputs"] = visual_inputs
    else:
        for i in range(batch_size):
            batch_kwargs[i]["visual_inputs"] = None

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    """Zero-pads a list of 1D tensors to equal length and stacks them into a 2D tensor.

    Args:
        sequences (List[torch.Tensor]): 
            List of 1D tensors to be padded. All tensors must have the same dimension (1D).
        side (str, optional): 
            Side to apply padding - either "left" (beginning of sequence) or "right" (end of sequence).
            Defaults to "left".

    Returns:
        torch.Tensor: 
            A 2D tensor of shape (num_sequences, max_length) containing padded and stacked sequences.

    Example:
        >>> seqs = [torch.tensor([1,2]), torch.tensor([3,4,5])]
        >>> zero_pad_sequences(seqs, "left")
        tensor([[0, 1, 2],
                [3, 4, 5]])
    """
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], data_processor: Optional[BaseDataProcessor], packing_samples=False) -> Experience:
    """Combines BufferItems into a batched Experience object.

    Processes individual experience items into a batch format. Sequences and masks are either:
    - Padded to equal length and stacked (when packing_samples=False)
    - Kept as variable-length lists (when packing_samples=True)

    Args:
        items (List[BufferItem]): 
            List of experience items to be batched together. All items must have matching
            keys in their 'info' dictionaries.
        packing_samples (bool, optional): 
            If True, maintains original sequence lengths without padding. Returns lists of tensors
            instead of padded tensors. Defaults to False.

    Returns:
        Experience: 
            A batched experience object containing:
            - Padded & stacked tensors (when packing_samples=False)
            - Lists of original-length tensors (when packing_samples=True)
            - Info values converted to tensors for scalar values

    Note:
        All BufferItems must have identical keys in their 'info' dictionaries. Scalar values in
        'info' are automatically converted to tensors.
    """
    kwargs = {}
    keys = (
        "sequences",
        "ref_log_probs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        # 'pixel_values',
        # 'image_flags'
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = [item.info[key] for item in items]
        if not (isinstance(vals[0], str) or key.startswith('dump')):
            vals = torch.tensor(vals)
        kwargs["info"][key] = vals

    if data_processor is not None:
        kwargs["visual_inputs"] = data_processor.make_input_batch([item.visual_inputs for item in items])
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, ref_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.ref_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.ref_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            ref_log_prob[:right_pad],
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, 
        sample_batch_size: int, 
        data_processor: Optional[BaseDataProcessor] = None, 
        limit: int = 0, 
        cpu_offload: bool = True, 
        packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.data_processor = data_processor
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience, self.data_processor)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.data_processor, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.data_processor, self.packing_samples)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
