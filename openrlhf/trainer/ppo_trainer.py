import os
import os.path
import json
from abc import ABC
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from filelock import FileLock

import time
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss, KLPenaltyLoss
# from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler, RandomSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer, DATA_PROCESSOR_MAP


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        processor: Optional[Callable[[Any], Dict]] = None,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.processor = processor
        self.data_processor = None
        # for vlm critic model, not provice processor.
        if (self.args.train_vlm or self.args.train_agent) and processor is not None:
            self.data_processor = DATA_PROCESSOR_MAP[type(processor)](processor)
            self.tokenizer = self.data_processor.tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor

        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()
        self.kl_loss = self.strategy.args.kl_loss_coef > 0
        self.kl_penalty_type = self.strategy.args.kl_penalty_type
        self.kl_threshold_type = self.strategy.args.kl_threshold_type
        if self.kl_loss:
            self.kl_loss_fn = KLPenaltyLoss(kl_type=self.strategy.args.kl_penalty_type)
        self.normalize_advantage = not self.strategy.args.not_normalize_advantage

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)
        
        
        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            self.data_processor,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, self.data_processor, buffer_limit, buffer_cpu_offload, packing_samples
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
        trained_steps=0, # unused here
    ) -> None:
        if args.train_agent:
            num_rollouts_per_episodes = len(prompts_dataloader)
        else:
            num_rollouts_per_episodes = (
                num_update_steps_per_episodes
                * args.train_batch_size
                // args.max_epochs
                // args.rollout_batch_size
                // args.n_samples_per_prompt
            )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler) \
                or isinstance(self.prompts_dataloader.sampler, RandomSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                experience_list = self.experience_maker.make_experience_list(rand_prompts, **self.generate_kwargs)
                
                for i, experience in enumerate(experience_list):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                # TODO: check this
                # torch.cuda.empty_cache() 
                if self.normalize_advantage:
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()
                # TODO: check this
                # torch.cuda.empty_cache()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {
                    "trained_steps": steps,
                    "consumed_samples": steps * args.rollout_batch_size
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train_agent:
            assert self.replay_buffer.sample_batch_size == 1, "only support batch size 1"
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            # shuffle=False if self.strategy.ring_attn_group is not None else True,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                # import pdb; pdb.set_trace()
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status_max = {
                        "kl_max": status.pop("kl_max", 0),
                        "kl_min": status.pop("kl_min", 0),
                        "kl_seq_max": status.pop("kl_seq_max", 0),
                        "kl_seq_min": status.pop("kl_seq_min", 0),
                    }
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status_max = self.strategy.all_reduce(status_max, op="max")
                    status["kl"] /= status["response_length"]
                    status.update(status_max)

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "loss": status["loss"],
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "kl_loss": status["kl_loss"] if "kl_loss" in status else 0,
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_max = {
                "kl_max": status_list[0].pop("kl_max", 0),
                "kl_min": status_list[0].pop("kl_min", 0),
                "kl_seq_max": status_list[0].pop("kl_seq_max", 0),
                "kl_seq_min": status_list[0].pop("kl_seq_min", 0),
            }
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    if k in ["kl_max", "kl_min", "kl_seq_max", "kl_seq_min"]:
                        status_max[k] = max(status_max[k], v)
                    else:
                        status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
            status_mean.update(status_max)
            # NOTE: grad norm should be after step
            # TODO: but if multiple updates per rollout, here only get the last step
            grad_norm = self.actor.model.get_global_grad_norm()
            status_mean["grad_norm"] = grad_norm.item() if grad_norm is not None else 0
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, global_steps)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience, global_steps:int) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            ref_log_probs = torch.cat(experience.ref_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            visual_inputs = experience.visual_inputs
            # pad seq makes the sequence a multiple of ring_attention_size.
            # if self.strategy.ring_attn_group is not None:
            #     pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
            #         sequences, attention_mask, num_actions, packed_seq_lens, self.strategy.ring_attn_group
            #     )
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            ref_log_probs = experience.ref_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            visual_inputs = experience.visual_inputs
            
        is_padding = experience.info.pop("is_padding", None)
        is_padding = is_padding[0] if is_padding is not None else False

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            # ring_attn_group=self.strategy.ring_attn_group,
            # logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs
        )
        # unpad sequence ensures that pad tokens do not contribute to the loss calculation.
        # if self.strategy.ring_attn_group is not None:
        #     assert pad_len is not None
        #     sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, _, _ = unpad_sequences(
        #         pad_len=pad_len,
        #         sequences=sequences,
        #         attention_mask=attention_mask,
        #         num_actions=num_actions,
        #         packed_seq_lens=packed_seq_lens,
        #         action_log_probs=action_log_probs,
        #         ring_attn_group=self.strategy.ring_attn_group,
        #     )

        # loss function
        # print(f"before actor loss, advantages: ", advantages)
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.kl_loss:
            scale_threshold = None
            if self.kl_threshold_type == "advantage":
                scale_threshold = advantages.abs() / self.args.kl_loss_coef
            kl_loss = self.kl_loss_fn(
                action_log_probs,
                ref_log_probs,
                action_mask=experience.action_mask,
                scale_threshold=scale_threshold,
            ) 
        else:
            kl_loss = 0

        if 'learn_mask' in experience.info:
            actor_loss = experience.info['learn_mask'] * actor_loss
            print(f'actor_loss: {actor_loss}')
            learn_mask = experience.info['learn_mask']
            print(f'learn_mask: {learn_mask}')

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        

        # Release the reference to output in a timely manner to avoid
        #  the lifetime of output lasting throughout the entire backward process.
        del output

        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.args.kl_loss_coef
        if is_padding:
            loss = loss * 0. # zero loss for padding samples
            actor_loss = actor_loss * 0. # for logging
            kl_loss = kl_loss * 0. # for logging
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # grad_norm = self.actor.model.get_global_grad_norm() not valid in the micro steps of gradient accumulation
        # status
        status = {
            "loss": loss.item(),
            "policy_loss": actor_loss.item(), 
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
            # "grad_norm": grad_norm.item() if grad_norm is not None else 0
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        if self.kl_loss:
            status['kl_loss'] = kl_loss.item()
        dump_infos = defaultdict(list)
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            if k.startswith("dump"):
                dump_infos[k].append(v)
            elif k in ["kl_max", "kl_min", "kl_seq_max", "kl_seq_min"]:
                status[k] = v.max().item()
            else:
                status[k] = v.mean().item()

        # 将 dump_infos 转换为列表
        dump_info_list = []
        for i in range(len(next(iter(dump_infos.values())))):
            dump_info = {}
            for k, v in dump_infos.items():
                dump_info[k.removeprefix('dump/')] = v[i]
            dump_info['global_steps'] = global_steps
            dump_info['advantage'] = advantages.mean().item()
            dump_info_list.append(dump_info)

        # 保存 dump_info_list 到文件
        save_root = os.path.join(self.strategy.args.ckpt_path, "dump_info")
        os.makedirs(save_root, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        with open(os.path.join(save_root, f"dump_info_{global_steps}_rank{rank}.jsonl"), "a", encoding="utf-8") as f:
            for dump_info in dump_info_list:
                f.write(json.dumps(dump_info) + "\n")

        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)

            if self.strategy.is_rank_0():
                eval_status = self.evaluate(global_step)
                if self._wandb is not None:
                    logs = {
                        "eval/%s" % k: v 
                        for k, v in {
                            **eval_status,
                            "global_step": global_step,
                        }.items()
                    }
                    self._wandb.log(logs)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def evaluate(self, global_step:int):
        raise NotImplementedError

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.save_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if args.save_hf_model:
            output_dir = os.path.join(args.save_path, '_actor', tag, 'hf_model')
            self.strategy.save_model(
                model=self.actor.model,
                tokenizer=self.tokenizer,
                output_dir=output_dir
            )

        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.save_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
