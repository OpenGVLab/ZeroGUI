import os
import json
import argparse
import copy
import datetime
from tqdm import tqdm

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import SamplingParams

from openrlhf.env import create_env
from openrlhf.agent import create_agent
from openrlhf.trainer.ray.vllm_agent_engine import LLMRayAgent
from openrlhf.trainer.ray.utils import ray_noset_visible_devices, get_bundle_indices


def main(args):
    env, env_config = create_env(args)
    agent = create_agent(args)

    ray.init(ignore_reinit_error=True,)
    tensor_parallel_size = args.vllm_tensor_parallel_size
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    num_gpus = int(tensor_parallel_size == 1)
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(tensor_parallel_size)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
    )
    bundle_indices = get_bundle_indices(pg, 0, tensor_parallel_size)

    ray_agent = LLMRayAgent.options(
        num_cpus=num_gpus,
        num_gpus=num_gpus,
        scheduling_strategy=scheduling_strategy,
    ).remote(
        agent=agent,
        env=env, 
        env_config=env_config,
        model=args.pretrain,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        noset_visible_devices=ray_noset_visible_devices(),
        limit_mm_per_prompt={'image': args.num_input_image},
        save_dir=args.save_dir,
        num_gpus=1,
        bundle_indices=bundle_indices,
        distributed_executor_backend=distributed_executor_backend,
        worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        max_tokens=1024,
    )

    with open(args.data_path, 'r') as f:
        lines = f.readlines()
    task_metas = [json.loads(line) for line in lines]

    task_acc = []
    for task_meta in tqdm(task_metas):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_timestamp = timestamp + f"_{task_meta.get('example_id', '')}"
        outputs = ray.get(ray_agent.run_single_task.remote(task_meta=task_meta, sampling_params=sampling_params,
                                                           timestamp=save_timestamp))
        task_acc.append(outputs['reward'] if outputs['reward'] >= 0 else 0.0)
        dump_info = copy.deepcopy(task_meta)
        dump_info.update({"timestamp": timestamp, "acc": outputs['reward']})
        with open(os.path.join(args.save_dir, 'task_info.jsonl'), 'a') as f:
            f.write(json.dumps(dump_info) + '\n')

    avg_success_rate = sum(task_acc) / len(task_acc)
    print(f"Average success rate: {avg_success_rate}")
    env.close()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval
    parser.add_argument("--data_path", type=str, default='./data/osworld_test_all.jsonl')
    parser.add_argument("--save_path", type=str, default='./results')
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )

    # env
    parser.add_argument("--env_type", type=str, default=None)
    parser.add_argument("--env_url", type=str, default=None)
    parser.add_argument("--env_port", type=int, default=None)
    parser.add_argument("--env_manager_port", type=int, default=None)
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--agent_max_steps", "--max_steps", dest='agent_max_steps', type=int, default=15)
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--use_llm_evaluator", action="store_true", default=False)
    parser.add_argument("--test_task_llm_eval", action="store_true", default=False)

    # agent
    parser.add_argument("--pretrain", type=str, default=None, help="model name or path")
    parser.add_argument("--agent_type", type=str, default='uitars')
    parser.add_argument("--agent_action_space", type=str, default='computer')
    parser.add_argument("--agent_prompt_language", type=str, default='Chinese')
    parser.add_argument("--num_history", type=int, default=5)
    parser.add_argument("--num_input_image", type=int, default=5)

    # sampling
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=1.0)

    args = parser.parse_args()
    model_name = os.path.basename(args.pretrain)
    args.save_dir = os.path.join(args.save_path, 'eval', model_name)
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
