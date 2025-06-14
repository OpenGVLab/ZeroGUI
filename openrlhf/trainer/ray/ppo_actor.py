import itertools
import math
import os
import socket
from typing import Callable, Dict, List
from copy import deepcopy

import deepspeed
import ray
import torch
torch.cuda.memory._set_allocator_settings('expandable_segments:False')
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer, dapo_agent_trainer_fit
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker, AgentExperienceMaker
from openrlhf.trainer.ray.vllm_agent_engine import batch_vllm_engine_call
from openrlhf.utils import blending_datasets, get_tokenizer, get_vl_processor, simple_load_datasets, TaskMetaDataset
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.distributed_sampler import RandomSampler
from vllm import SamplingParams

from .launcher import BasePPORole
from .utils import get_physical_gpu_id


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        if self.strategy.args.train_agent:
            exp_maker_cls = AgentExperienceMaker
        else:
            exp_maker_cls = RemoteExperienceMaker
        self.experience_maker = exp_maker_cls(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        # init evaluators
        self.evaluators = []

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True
        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
                
            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch_dist_barrier_and_cuda_sync()
        status = {}

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(super().ppo_train(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                torch_dist_barrier_and_cuda_sync()
                self._broadcast_to_vllm()

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch_dist_barrier_and_cuda_sync()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch_dist_barrier_and_cuda_sync()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience, global_steps)

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=count == num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch_dist_barrier_and_cuda_sync()

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch_dist_barrier_and_cuda_sync()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if self.critic_train_remote:
            ref = self.critic.save_checkpoint.remote(tag)
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
        # wait
        if self.critic_train_remote:
            ray.get(ref)
        torch_dist_barrier_and_cuda_sync()


    def evaluate(self, global_step:int):
        args = self.strategy.args
        status = {}
        for evaluator in self.evaluators:
            evaluator.refresh()
            self.strategy.print(f"start evaluation on {evaluator.name}")
            save_path = os.path.join(args.ckpt_path, 'evaluation', f"step_{global_step}", evaluator.name)

            def generate(prompts, system=None, temperature=0.6, top_p=0.95, max_tokens=32768):
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                
                def process_prompt(input_prompt):
                    """Process a single prompt with template or chat formatting"""
                    if args.apply_chat_template:
                        messages = [{"role": "user", "content": input_prompt}]
                        if system:
                            messages.insert(0, {"role": "system", "content": system})
                        return self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        assert args.input_template is not None
                        return args.input_template.format(input_prompt)

                full_prompts = []
                for prompt in prompts:
                    # Handle both string and dictionary format prompts
                    if isinstance(prompt, str):
                        processed = process_prompt(prompt)
                    else:
                        processed = deepcopy(prompt)
                        processed['prompt'] = process_prompt(processed['prompt'])
                    
                    full_prompts.append(processed)
                
                output_refs = []
                batch_size = (len(full_prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
                for i, llm in enumerate(self.vllm_engines):
                    prompts = full_prompts[i * batch_size : (i + 1) * batch_size]
                    if prompts:
                        output_refs.append(llm.generate.remote(prompts, sampling_params))
                outputs = sum(ray.get(output_refs), [])

                texts = [x.outputs[0].text for x in outputs]
                return texts

            scores = evaluator.run(
                model=generate, 
                save_path=save_path,
                max_tokens=args.generate_max_len
            )
            
            for key, value in scores.items():
                status[f"{evaluator.name}/{key}"] = value

        return status

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            args=args,
        )
        strategy.print(actor)

        # configure tokenizer
        if args.train_vlm or args.train_agent:
            self.processor = get_vl_processor(
                pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = get_tokenizer(
                pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        # self._num_image_token is used to initialize datasets
        self._num_image_token = getattr(actor.model, 'num_image_token', 256)
        self.prepare_datasets()

        # configure scheduler
        if args.num_train_steps is not None:
            max_steps = args.num_train_steps
            self.num_update_steps_per_episodes = -1 # TODO: no fixed value for `num_update_steps_per_episodes`
        else:
            if args.train_agent:
                self.num_update_steps_per_episodes = len(self.prompts_dataset) * args.n_samples_per_prompt * args.agent_max_steps \
                                                     // args.train_batch_size * args.max_epochs
            else:
                self.num_update_steps_per_episodes = (
                    len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
                )
            max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            getattr(args, "actor_lr_schedule", "cosine_with_min_lr"),
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        self.trained_steps = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path, args.ckpt_tag)
            self.consumed_samples = states["consumed_samples"]
            self.trained_steps = states.get("trained_steps", 0)
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}, trained_steps: {self.trained_steps}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        datasets = args.prompt_data
        if args.simple_load_dataset:
            prompts_data = simple_load_datasets(datasets, strategy=strategy)
        else:
            prompts_data = blending_datasets(
                datasets,
                args.prompt_data_probs,
                strategy,
                args.seed,
                max_count=args.max_samples,
                return_eval=False,
                train_split=args.prompt_split,
            )
            prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        if args.train_agent:
            self.prompts_dataset = TaskMetaDataset(prompts_data)
        else:
            self.prompts_dataset = PromptDataset(
                prompts_data, self.tokenizer, strategy, input_template=args.input_template
            )

        shuffle = not args.no_shuffle_train_dataset
        if args.task_sampling_distributed or args.task_group_distributed:
            # distribute the sampling of a task to different ranks
            # so sample the same task for all ranks
            sampler = RandomSampler(
                self.prompts_dataset,
                shuffle=shuffle,
                seed=args.seed,
            )
            batch_size = args.rollout_batch_size
        else:
            sampler = None
            batch_size = args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size)
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, 
            batch_size=batch_size,
            pin_memory=True, 
            shuffle=shuffle,
            collate_fn=lambda x:x,
            sampler=sampler,
        )

        if args.pretrain_data:
            raise NotImplementedError("Please unset `--pretrain_data` as it has not been supported yet.")
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            processor=self.processor, 
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # patch
        if args.train_agent and args.use_dapo_trainer:
            trainer.fit = dapo_agent_trainer_fit.__get__(trainer, ActorPPOTrainer)

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "wake_up")
            torch_dist_barrier_and_cuda_sync()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch_dist_barrier_and_cuda_sync()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
            self.trained_steps,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )
