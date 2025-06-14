import os
import json
import ray
import datetime
import queue
from collections import defaultdict
from typing import Any, List

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.env import create_env
from openrlhf.agent import create_agent
from openrlhf.trainer.ray.utils import ray_noset_visible_devices, get_bundle_indices


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayAgent():
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        self.env = kwargs.pop("env")
        self.env_config = kwargs.pop("env_config")
        self.agent = kwargs.pop("agent")
        self.save_dir= kwargs.pop("save_dir")
        
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        import vllm
        
        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def run_single_task(self, task_meta, sampling_params, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            timestamp += "_" + task_meta.get('example_id', "")
        if self.save_dir is not None:
            task_save_dir = os.path.join(self.save_dir, timestamp)
            os.makedirs(task_save_dir, exist_ok=True)
        # runtime_logger = setup_logger(example, example_result_dir)
        self.agent.reset()

        task_config = self.env.get_task_config(**task_meta)
        instruction = task_config["instruction"]
        obs = self.env.reset(task_config=task_config) # 查看obs
        if obs is None:
            with open(os.path.join(task_save_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"Env reset failed.")
            return {
                "steps": [],
                "num_steps": 0,
                "reward": -1,
                "is_valid": False,
            }
        if self.save_dir is not None:
            with open(os.path.join(task_save_dir, f"step_0.png"), "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(task_save_dir, "config.json"), "a") as f:
                json.dump(task_config, f, indent=4)

        done = False
        is_step_error = False
        step_idx = 0
        # env.controller.start_recording()
        steps = []
        traj_for_eval = {"screenshots": [], "actions": []}
        traj_for_eval["screenshots"].append(obs['screenshot'])
        while not done and step_idx < self.env_config['max_steps']:
            inputs = self.agent.get_model_inputs(instruction, obs)
            outputs = self.generate(prompts=[inputs], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            actions = self.agent.parse_action(response)

            steps.append({
                'input_text': inputs['prompt'],
                'output_text': response,
                'input_ids': outputs[0].prompt_token_ids,
                'output_ids': list(outputs[0].outputs[0].token_ids),
                'image': inputs['multi_modal_data']['image'],
            })
            # save model input & output texts whether or not parsing action is successful
            if self.save_dir is not None:
                with open(os.path.join(task_save_dir, "model_output.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "parsed_action": actions,
                        "input_text": inputs['prompt'],
                        "output_text": response,
                    }, ensure_ascii=False))
                    f.write("\n")

            for action in actions:
                obs, reward, done, info = self.env.step(action) # pause=self.env_config['pause'])
                # deal with step error
                if (obs is None) and done:
                    is_step_error = True
                    break

                # If parsing action succeeds, execute the action and
                # Save screenshot and trajectory information
                traj_for_eval["screenshots"].append(obs['screenshot'])
                traj_for_eval["actions"].append(action)
                if self.save_dir is not None:
                    with open(os.path.join(task_save_dir, f"step_{step_idx + 1}.png"),
                            "wb") as _f:
                        _f.write(obs['screenshot'])
                
                    with open(os.path.join(task_save_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({
                            "step_num": step_idx + 1,
                            "action": action,
                            "reward": float(reward),
                            "done": done,
                            "info": info,
                            "screenshot_file": f"step_{step_idx + 1}.png",
                            "input_text": inputs['prompt'],
                            "output_text": response,
                        }, ensure_ascii=False))
                        f.write("\n")
                if done:
                    # logger.info("The episode is done.")
                    break
            step_idx += 1

        if is_step_error:
            # TODO: if env.step is failed, ignore this trajectory now
            # but sometimes the VM is closed caused by the predicted action, what about the reward?
            result = -1
            eval_outputs = {"reward": -1}
        else:
            eval_outputs = self.env.evaluate(task_config, traj_for_eval)
            result = eval_outputs["reward"]
        if self.save_dir is not None:
            with open(os.path.join(task_save_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"{result}\n")
            with open(os.path.join(task_save_dir, "eval.json"), "w") as f:
                json.dump(eval_outputs, f, indent=4)

        return_dict = {
            "steps": steps,
            "num_steps": len(steps),
            "reward": float(result),
            "is_valid": result >= 0,
        }
        return return_dict

def create_vllm_agent_engines(
    args,
    num_engines: int,
    pretrain: str,
    tensor_parallel_size: int,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    save_dir: str = None,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    **kwargs
):
    vllm_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())
    
    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)

        env, env_config = create_env(args, env_idx=i)
        agent = create_agent(args)

        vllm_engines.append(
            LLMRayAgent.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                agent=agent, 
                env=env, 
                env_config=env_config,
                model=pretrain,
                noset_visible_devices=ray_noset_visible_devices(),
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                save_dir=save_dir,
                worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                distributed_executor_backend=distributed_executor_backend,
                full_determinism=full_determinism,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                **kwargs
            )
        )
        
    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines

def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))
    return ray.get(refs)
