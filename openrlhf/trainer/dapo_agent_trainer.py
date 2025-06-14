import io
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from .ppo_trainer import PPOTrainer
from openrlhf.utils.distributed_sampler import DistributedSampler, RandomSampler
from openrlhf.utils.distributed_util import torch_dist_barrier_and_cuda_sync


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def data_gather_redistribute(steps):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # split redistribute / local
    device = torch.cuda.current_device()
    num_steps = torch.tensor([len(steps)], dtype=torch.long, device=device)
    all_num_steps = [torch.zeros_like(num_steps) for _ in range(world_size)]
    torch.distributed.all_gather(all_num_steps, num_steps)
    all_num_steps = torch.cat(all_num_steps)
    min_num_steps = all_num_steps.min().item()
    local_steps, steps = steps[:min_num_steps], steps[min_num_steps:]

    # Image -> bytes
    for step in steps:
        images = step["image"]
        step["image"] = [image_to_bytes(img) for img in images]

    # gather
    steps_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(steps_list, steps)
    steps = sum(steps_list, [])

    # redistribute
    selected_num = len(steps) // world_size
    steps = steps[rank * selected_num : (rank + 1) * selected_num]

    # bytes -> Image
    for step in steps:
        images = step["image"]
        step["image"] = [bytes_to_image(img) for img in images]

    steps = local_steps + steps
    return steps


def dapo_agent_trainer_fit(
    self: PPOTrainer,
    args,
    prompts_dataloader,
    pretrain_dataloader,
    consumed_samples=0,
    num_update_steps_per_episodes=1,
    trained_steps=0,
) -> None:
    assert args.train_agent, "only support agent mode"
    num_rollouts_per_episodes = len(prompts_dataloader)

    # get eval and save steps
    if args.eval_steps == -1:
        args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
    if args.save_steps == -1:
        args.save_steps = float("inf")  # do not save ckpt

    self.prompts_dataloader = prompts_dataloader
    self.pretrain_dataloader = pretrain_dataloader

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # control the rollout sampling and collection
    rollout_target_size = args.rollout_target_size # target number of samples in a rollout
    num_updates_per_rollout = rollout_target_size // args.train_batch_size
    local_target_size = rollout_target_size // world_size
    num_gen_batches = 0
    steps_collected = []

    # Restore step and start_epoch
    steps = trained_steps + 1
    num_global_steps = args.num_train_steps // args.max_epochs // num_updates_per_rollout
    start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
    consumed_samples_start_ep = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
    progress_bar = tqdm(
        total=num_global_steps,
        initial=trained_steps, 
        desc="Training Progress",
        disable=not self.strategy.is_rank_0()
    )

    for episode in range(start_episode, args.num_episodes):
        if isinstance(self.prompts_dataloader.sampler, DistributedSampler) \
            or isinstance(self.prompts_dataloader.sampler, RandomSampler):
            self.prompts_dataloader.sampler.set_epoch(
                episode, consumed_samples=0 if episode > start_episode else consumed_samples_start_ep
            )
        pbar = tqdm(
            range(self.prompts_dataloader.__len__()),
            desc=f"Episode [{episode + 1}/{args.num_episodes}]",
            disable=not self.strategy.is_rank_0(),
        )

        for batch_prompts in self.prompts_dataloader:
            # group distributed
            if args.task_group_distributed:
                n_groups = args.num_distributed_groups
                group_index = rank % n_groups
                group_bs = len(batch_prompts) // n_groups
                batch_prompts = batch_prompts[group_index * group_bs : (group_index + 1) * group_bs]

            # vLLM wakeup when vllm_enable_sleep
            if self.args.vllm_enable_sleep:
                torch.cuda.empty_cache()
                torch_dist_barrier_and_cuda_sync()
                from openrlhf.trainer.ray.vllm_agent_engine import batch_vllm_engine_call
                batch_vllm_engine_call(self.vllm_engines, "wake_up")
                torch_dist_barrier_and_cuda_sync()

            # sample trajectories
            new_steps_collected = []
            for task_meta in tqdm(
                batch_prompts,
                desc="Sampling tasks",
                disable=not self.strategy.is_rank_0(),
            ):
                trajectories, info = self.experience_maker.sample_tractory(task_meta, **self.generate_kwargs)

                # DAPO dynamic sampling
                if args.dapo_dynamic_sampling and (info["acc"] == 0 or info["acc"] == 1):
                    continue # skip task (global) acc = 0 or 1
                # TODO: balance positive / negative samples here? 

                # expand as steps
                for trajectory in trajectories:
                    if trajectory['is_valid']:
                        new_steps_collected.extend(trajectory['steps'])

            # vLLM offload when vllm_enable_sleep
            if self.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
                torch.cuda.empty_cache()
                torch_dist_barrier_and_cuda_sync()

            # gather and re-distribute for better efficiency
            if self.args.data_gather_redistribute:
                new_steps_collected = data_gather_redistribute(new_steps_collected)
            steps_collected += new_steps_collected

            consumed_samples += args.rollout_batch_size
            num_gen_batches += 1
            pbar.update()

            # determine whether continues sampling
            device = torch.cuda.current_device()
            collected_samples = torch.tensor([len(steps_collected)], dtype=torch.long, device=device)
            all_collected_samples = [torch.zeros_like(collected_samples) for _ in range(world_size)]
            torch.distributed.all_gather(all_collected_samples, collected_samples)
            all_collected_samples = torch.cat(all_collected_samples)
            if torch.any(all_collected_samples < local_target_size):
                print(f'collected samples: {len(steps_collected)}, target size: {local_target_size}')
                max_num_gen_batches = args.max_num_gen_batches
                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                    print(f'{num_gen_batches=}. Keep generating...')
                    continue
                else:
                    raise ValueError(
                        f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                    )

            # vLLM offload when vllm_enable_sleep
            # if self.args.vllm_enable_sleep:
            #     batch_vllm_engine_call(self.vllm_engines, "sleep")
            #     torch.cuda.empty_cache()
            #     torch_dist_barrier_and_cuda_sync()

            if not args.only_sample:
                # generate and sample experience list
                steps_collected = np.random.choice(steps_collected, local_target_size, replace=False)
                experience_list = self.experience_maker.make_experience_step_list(steps_collected)

                for i, experience in enumerate(experience_list):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                # torch.cuda.empty_cache()
                if self.normalize_advantage:
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()
                # torch.cuda.empty_cache()
            else:
                status = {}
            
            # renew rollout control
            status["num_gen_batches"] = num_gen_batches
            num_gen_batches = 0
            steps_collected = []

            if "kl" in status:
                self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
            pbar.set_postfix(status)

            # logs/checkpoints
            client_states = {
                "trained_steps": steps,
                "consumed_samples": consumed_samples,
            }
            self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

            if steps >= num_global_steps:
                if self._wandb is not None and self.strategy.is_rank_0():
                    self._wandb.finish()
                if self._tensorboard is not None and self.strategy.is_rank_0():
                    self._tensorboard.close()
                return

            steps = steps + 1
            progress_bar.update(1)
