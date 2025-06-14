from datetime import datetime
import os
import io
from typing import Optional
import json
from pathlib import Path
import random
import string

from torch.utils.data import Dataset
from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
import json
import torch
from pathlib import Path
from tqdm import tqdm

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def randomID(k=8):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=k))

def get_vl_processor(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # TODO: Maybe better max_pixels set methods for other vl model
    # min_pixels = int(os.getenv("MIN_PIXELS", 4*28*28))
    # max_pixels = int(os.getenv("MAX_PIXELS", 640*28*28))
    # processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast, min_pixels=min_pixels, max_pixels=max_pixels)
    processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, trust_remote_code=True,  use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=trust_remote_code, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def mp_run(work_fn, vars, num_workers=32, total=None, desc=None, keep_order=False, mode='process'):
    """
    Execute tasks in parallel using either multiprocessing or multithreading.

    Args:
        work_fn: The function to apply to each element in vars.
        vars: A list or tuple of variables to process.
        num_workers: The number of parallel workers (threads or processes).
        total: The total number of tasks (if None, it's inferred from vars).
        desc: Description for the progress bar.
        keep_order: If True, results will be returned in the order of vars.
        mode: 'process' for multiprocessing, 'thread' for multithreading.

    Returns:
        List of results.
    """
    
    if total is None and isinstance(vars, (list, tuple)):
        total = len(vars)

    res = []
    
    try:
        if mode == 'process':
            # Use multiprocessing
            with multiprocessing.Pool(num_workers) as pool:
                map_fn = pool.imap if keep_order else pool.imap_unordered
                with tqdm(total=total, desc=desc) as pbar:
                    for r in map_fn(work_fn, vars):
                        pbar.update(1)
                        res.append(r)

        elif mode == 'thread':
            # Use multithreading
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                with tqdm(total=total, desc=desc) as pbar:
                    if keep_order:
                        for r in executor.map(work_fn, vars):
                            pbar.update(1)
                            res.append(r)
                    else:
                        futures = [executor.submit(work_fn, v) for v in vars]
                        for future in futures:
                            r = future.result()
                            pbar.update(1)
                            res.append(r)
                            
    except KeyboardInterrupt:
        print("\n中断信号收到。正在终止任务并收集已完成的结果...")
        if mode == 'process':
            pool.terminate()
        elif mode == 'thread':
            executor.shutdown(wait=False)
    
    return res

def read_jsonl(file, client=None):
    if client is None or not file.startswith('s3://'):
        assert Path(file).is_file(), file
        dataset = open(file, buffering=int(32e6)).readlines()
    else:
        data_str = client.get(file).decode()
        data_io = io.StringIO(data_str)
        dataset = data_io.readlines()
    outputs = []
    for i, line in enumerate(dataset):
        line = line.rstrip()
        if line:
            try:
                outputs.append(json.loads(line))
            except Exception as e:
                print(f'i={i}, line={line}')
                # raise e
    return outputs

def enable_pytorch_expandable_segments(with_max_memory_fraction: Optional[int] = None):
    if torch.__version__ >= "2.1.0":
        _expandable_segments_conf = "expandable_segments:True"
        _alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", None)
        if _alloc_conf is None:
            _alloc_conf = _expandable_segments_conf
        elif "max_split_size_mb" not in _alloc_conf:
            _alloc_conf = _alloc_conf + "," + _expandable_segments_conf

        torch.cuda.memory._set_allocator_settings(_alloc_conf)
        print("Enable expandable_segments!", flush=True)
    else:
        print("To support the 'expandable_segments' configuration, please upgrade torch to version 2.1.0.")

    if with_max_memory_fraction is not None:
        print(f"Set pytorch per process memory fraction: {with_max_memory_fraction}", flush=True)
        torch.cuda.set_per_process_memory_fraction(with_max_memory_fraction)

def get_generation_cls(config):
    model_type = config.model_type
    model_arch = AutoModel._model_mapping[type(config)].__name__
    if model_arch.endswith("ForCausalLM") or \
    model_arch.endswith("ForConditionalGeneration"):
        return AutoModel._model_mapping[type(config)]
    elif model_arch.endswith("Model"):
        possible_arch = [model_arch.replace("Model", "ForCausalLM"), model_arch.replace("Model", "ForConditionalGeneration")]
        import importlib
        module = importlib.import_module(f".models.{model_type}.modeling_{model_type}",package="transformers")
        for arch in possible_arch:
            model_cls = getattr(module, arch, None)
            if model_cls is not None:
                return model_cls
        raise ValueError(f"Cannot find ForCausalLM or ForConditionalGeneration class for {model_arch}")
    else:
        raise ValueError(f"Unexpected model architecture {model_arch}")


class TaskMetaDataset(Dataset):
    def __init__(self, task_metas):
        super().__init__()
        self.task_metas = task_metas

    def __len__(self):
        return len(self.task_metas)

    def __getitem__(self, index):
        return self.task_metas[index]


def simple_load_datasets(datasets, strategy=None):
    datasets = datasets.split(",")
    train_data_list = []

    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")
        ext = os.path.splitext(dataset)[-1]

        if ext == ".jsonl":
            with open(dataset, 'r') as f:
                lines = f.readlines()
            data = [json.loads(line) for line in lines]
            train_data_list.append(data)
            strategy.print(f"loaded: {dataset} with data_files={dataset}")
        else:
            raise NotImplementedError

    # merge datasets
    train_dataset = sum(train_data_list, [])
    return train_dataset