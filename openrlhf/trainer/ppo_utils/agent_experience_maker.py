import os
import json
import ray
import torch
import time
import datetime
import random
from copy import deepcopy
from tqdm import tqdm
from vllm import SamplingParams
from typing import List, Optional, Tuple, Union, Dict

from openrlhf.models.utils import compute_approx_kl
from .experience_maker import RemoteExperienceMaker, Experience, Samples

from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync

class AgentExperienceMaker(RemoteExperienceMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: now follow packing mode, but do not pack multiple samples
        assert self.packing_samples, "Only support packing mode"

    @torch.no_grad()
    def make_experience_list(self, all_prompts, **generate_kwargs) -> List[Experience]:
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_agent_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch_dist_barrier_and_cuda_sync()
            
        args = self.strategy.args
        # group distributed
        if args.task_group_distributed:
            n_groups = args.num_distributed_groups
            group_index = torch.distributed.get_rank() % n_groups
            group_bs = len(all_prompts) // n_groups
            all_prompts = all_prompts[group_index * group_bs : (group_index + 1) * group_bs]

        # NOTE: iterate over tasks
        experiences = []
        traj_list = []
        for task_meta in tqdm(
            all_prompts,
            desc="Sampling tasks",
            disable=not self.strategy.is_rank_0(),
        ):
            trajectories, info = self.sample_tractory(task_meta, **generate_kwargs)
            traj_list.append(trajectories)
        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()
        for trajectories in traj_list:
            experiences.extend(self.make_experience(trajectories, step_padding=True))

        # TODO: expand var-len steps of all trajectories into a single list
        # loss weight may be affected. ways to solve this?
        return experiences
            
    @torch.no_grad()
    def sample_tractory(self, task_meta, **kwargs):
        args = self.strategy.args

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if args.task_group_distributed:
            n_groups = args.num_distributed_groups
            group_size = world_size // n_groups
            group_index = rank % n_groups
            group_rank = rank // n_groups

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            # llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
            llms = [self.vllm_engines[rank // (world_size // len(self.vllm_engines))]] # even distribute different groups
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 512),
            min_tokens=kwargs.get("min_new_tokens", 1),
            frequency_penalty=kwargs.get("frequency_penalty", 0),
            # skip_special_tokens=kwargs.get("skip_special_tokens", True),
            # include_stop_str_in_output=True,
        )

        # all_metas = [task_meta] * args.n_samples_per_prompt
        # batch_size = (len(all_metas) + len(llms) - 1) // len(llms)

        if args.task_sampling_distributed:
            # distribute the sampling of a task to different ranks
            assert args.n_samples_per_prompt % world_size == 0
            n_samples = args.n_samples_per_prompt // world_size
        elif args.task_group_distributed:
            assert args.n_samples_per_prompt % group_size == 0
            n_samples = args.n_samples_per_prompt // group_size
        else:
            n_samples = args.n_samples_per_prompt

        # NOTE: iterate over sampling times
        # Distribute requests to engines and collect responses to outputs
        refs = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # avoid all env reset at the same time
        sleep_time = (rank if not args.task_group_distributed else group_rank) / world_size * args.env_reset_sleep_range
        time.sleep(sleep_time + random.Random(rank + int(time.time())).random())
        for i in range(n_samples):
            llm_id = i % len(llms) # TODO: simple even distribution
            timestamp_sample = timestamp + f"_{task_meta.get('example_id', '')}_rk{rank}_sample{i}" # distinguish different ranks
            refs.append(
                llms[llm_id].run_single_task.remote(task_meta=task_meta, sampling_params=sampling_params,
                                                    timestamp=timestamp_sample)
            )
        outputs = ray.get(refs)

        # compute task acc
        torch.distributed.barrier()
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        rewards = [t["reward"] for t in outputs] # (n_samples, )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        # TODO: deal with unhandled env error
        valid_mask = rewards >= 0 # reward -1 means env failed

        if args.task_sampling_distributed or args.task_group_distributed:
            # gather the reward from all ranks
            all_rewards = [torch.zeros_like(rewards) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_rewards, rewards)
            all_rewards = torch.cat(all_rewards)
            all_valid_mask = [torch.zeros_like(valid_mask) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_valid_mask, valid_mask)
            all_valid_mask = torch.cat(all_valid_mask)

            if args.task_group_distributed:
                all_rewards = all_rewards.reshape(group_size, n_groups, n_samples)[:,group_index].reshape(-1)
                all_valid_mask = all_valid_mask.reshape(group_size, n_groups, n_samples)[:,group_index].reshape(-1)
                assert len(all_rewards) == len(all_valid_mask) == args.n_samples_per_prompt
        else:
            all_rewards = rewards
            all_valid_mask = valid_mask

        # acc = (all_rewards == 1).float().mean().item()
        acc = (all_rewards * all_valid_mask).mean().item()

        # GRPO
        if self.advantage_estimator in ["group_norm"]:
            if all_valid_mask.sum() > 0:
                mean = all_rewards[all_valid_mask].mean(0, keepdim=True)
                std = all_rewards[all_valid_mask].std(0, keepdim=True)
                advantages = (rewards - mean) / (std + 1e-8)
                advantages = advantages * valid_mask
            else:
                advantages = torch.zeros_like(rewards)
                std = torch.zeros((1, ), dtype=rewards.dtype, device=rewards.device)
            for i, trajectory in enumerate(outputs):
                trajectory["advantage"] = advantages[i].item()
                trajectory["reward_std"] = std.item()
                trajectory["global_acc"] = acc
                # assign all steps with the same reward
                for step in trajectory['steps']:
                    step['reward'] = trajectory['reward']
                    step['reward_std'] = trajectory['reward_std']
                    step['advantage'] = trajectory['advantage']
        else:
            raise NotImplementedError

        # task info
        info = {
            "timestamp": timestamp,
            "acc": acc,
        } # TODO: can add more info
        torch.distributed.barrier()
        if (args.task_sampling_distributed and rank == 0) \
            or (args.task_group_distributed and group_rank == 0) \
            or ((not args.task_sampling_distributed) and (not args.task_group_distributed)):
            time.sleep(rank)
            dump_info = deepcopy(task_meta)
            dump_info.update(info)
            with open(os.path.join(args.save_path, 'task_info.jsonl'), 'a') as f:
                f.write(json.dumps(dump_info) + '\n')

        return outputs, info

    @torch.no_grad()
    def make_experience(self, trajectories, step_padding=True):
        self.actor.eval()

        experiences = []
        for trajectory in tqdm(
            trajectories,
            desc="Make experience",
            disable=not self.strategy.is_rank_0(),
        ):
            steps = trajectory['steps']
            is_valid = trajectory['is_valid']

            # NOTE: pad to max step to keep the same number of samples and avoid NCCL stuck.
            # TODO: better solution?
            for step_idx in range(self.args.agent_max_steps):
                if step_idx < len(steps):
                    step = steps[step_idx]
                    is_padding = False
                else: # for padding
                    step = steps[0]
                    is_padding = True
                if not is_valid: # env failed
                    is_padding = True
            
                experience = self.make_experience_single_step(step)
                experience.info["is_padding"] = [is_padding]
                # NOTE: if not step padding, only add valid sample
                if step_padding or (not is_padding):
                    experiences.append(experience)

        self.actor.train()  # reset model state
        return experiences

    @torch.no_grad()
    def make_experience_step_list(self, steps):
        self.actor.eval()

        experiences = []
        for step in tqdm(
            steps,
            desc="Make experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experience = self.make_experience_single_step(step)
            experience.info["is_padding"] = [False]
            experiences.append(experience)

        self.actor.train()  # reset model state
        return experiences

    @torch.no_grad()
    def make_experience_single_step(self, step):
        args = self.strategy.args
        device = torch.cuda.current_device()
        advantage = step['advantage']
        reward = step['reward']
        std = step['reward_std']

        # only one sample per sequence
        input_ids = step['input_ids']
        output_ids = step['output_ids']
        sequences_cpu = torch.LongTensor(input_ids + output_ids).unsqueeze(0) # (1, L)
        sequences = sequences_cpu.to("cuda")
        attention_mask, attention_mask_cpu = torch.ones_like(sequences), torch.ones_like(sequences_cpu) 
        num_actions = [max(1, len(output_ids))]
        packed_seq_lens = [len(input_ids) + len(output_ids)]
        
        # get visual inputs
        visual_inputs_cpu = self.data_processor.processor(
            text=step['input_text'],
            images=step['image'],
            return_tensors="pt",
        ) # ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
        visual_inputs_cpu.pop("input_ids")
        visual_inputs_cpu.pop("attention_mask") # keep 'pixel_values', 'image_grid_thw'
        visual_inputs = {k: v.to("cuda") for k, v in visual_inputs_cpu.items()}
    
        # compute logits
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs # TODO: why not visual_inputs_cpu?
        )
        if args.colocate_actor_ref or args.colocate_all_models:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])
        action_log_probs = self.actor(
            sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs
        )
        ref_log_probs = ray.get(base_action_log_probs_ref).to(device)

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            ref_log_probs,
            action_mask=None,
            use_kl_estimator_k3=args.use_kl_estimator_k3,
        )
        
        # advantage
        advantage_tensor = torch.ones_like(action_log_probs) * advantage
        returns = deepcopy(advantage_tensor)

        # info for logging
        response_length = torch.tensor(num_actions, device=device, dtype=torch.float)
        total_length = torch.tensor(packed_seq_lens, device=device, dtype=torch.float)
        info = {
            # "is_padding": [is_padding],
            "kl": kl.mean().unsqueeze(0),
            "kl_max": kl.max().unsqueeze(0),
            "kl_min": -(kl.min()).unsqueeze(0),
            "kl_std": kl.std().unsqueeze(0),
            "kl_seq_max": kl.mean().unsqueeze(0),
            "kl_seq_min": -(kl.mean()).unsqueeze(0),
            "acc": (torch.tensor(reward) == 1).float().unsqueeze(0), # TODO: split trajectory into multiple steps, average acc may not be accurate 
            "reward": torch.tensor(reward).unsqueeze(0), # initial reward
            "reward_std": torch.tensor(std).unsqueeze(0),
            "response_length": response_length,
            "total_length": total_length,
            # "num_actions": num_actions, # unused, delete in `make_experience_list
            "return": torch.tensor(advantage).unsqueeze(0), # in `make_experience_list, sum of reward (advantage),
            "dump/rewards": [reward],
        }

        # TODO: a hack? need to use list for var-len sequence
        experience = Experience(
            sequences=[sequences.squeeze(0)],
            action_log_probs=[action_log_probs.squeeze(0)],
            values=None,
            returns=[returns.squeeze(0)],
            advantages=[advantage_tensor.squeeze(0)],
            attention_mask=None, # set to None in `make_experience`
            action_mask=None,
            info=info,
            # kl=[kl.squeeze(0)],
            kl=None, # unused, delete in `make_experience_list
            ref_log_probs=[ref_log_probs.squeeze(0)],
            visual_inputs=visual_inputs,
        )
        return experience.to_device("cpu")
