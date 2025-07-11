import os
import json
import argparse
from datetime import datetime
from typing import List

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
    create_vllm_agent_engines,
)
from openrlhf.utils import get_strategy


# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)


def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.task_sampling_distributed:
        assert (
            args.n_samples_per_prompt % actor_world_size == 0
        ), f"n_samples_per_prompt must be divisible by actor_world_size, got {args.n_samples_per_prompt} and {actor_world_size}"
    elif args.task_group_distributed:
        assert (
            args.rollout_batch_size % args.num_distributed_groups == 0 and actor_world_size % args.num_distributed_groups == 0
        ), f"rollout_batch_size and actor_world_size must be divisible by num_distributed_groups, got {args.rollout_batch_size}, {actor_world_size} and {args.num_distributed_groups}"
        assert (
            args.n_samples_per_prompt % (actor_world_size // args.num_distributed_groups) == 0
        ), f"n_samples_per_prompt must be divisible by group size, got {args.n_samples_per_prompt} and {actor_world_size // args.num_distributed_groups}"
    else:
        assert (
            args.rollout_batch_size % actor_world_size == 0
        ), f"rollout_bach_size must be divisible by actor_world_size, got {args.rollout_batch_size} and {actor_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

    if args.vllm_num_engines > 0:
        assert (
            actor_world_size % args.vllm_num_engines == 0 or args.vllm_num_engines % actor_world_size == 0
        ), f"actor_world_size must be divisible by vllm_num_engines, got {actor_world_size} and {args.vllm_num_engines}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"


def train(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
    # pg = None
    # if args.colocate_actor_ref or args.colocate_all_models:
    #     assert (
    #         args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
    #     ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

    #     bundles = [
    #         {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
    #         for _ in range(args.actor_num_nodes)
    #     ]
    #     pg = placement_group(bundles, strategy="STRICT_SPREAD")
    #     ray.get(pg.ready())

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        if args.colocate_all_models:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        if args.train_agent:
            kwargs = {}
            if args.save_trajectory:
                kwargs['save_dir'] = os.path.join(args.save_path, 'trajectory')
            if args.num_input_image is not None:
                kwargs['limit_mm_per_prompt'] = {'image': args.num_input_image}
            vllm_engines = create_vllm_agent_engines(
                args,
                num_engines=args.vllm_num_engines,
                pretrain=args.pretrain,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                seed=args.seed,
                full_determinism=args.full_determinism,
                enable_prefix_caching=args.enable_prefix_caching,
                enforce_eager=args.enforce_eager,
                max_model_len=max_len,
                num_total_actors=args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
                shared_pg=pg if args.colocate_all_models else None,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_enable_sleep=args.vllm_enable_sleep,
                **kwargs
            )
        else:
            vllm_engines = create_vllm_engines(
                num_engines=args.vllm_num_engines,
                pretrain=args.pretrain,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                seed=args.seed,
                full_determinism=args.full_determinism,
                enable_prefix_caching=args.enable_prefix_caching,
                enforce_eager=args.enforce_eager,
                max_model_len=max_len,
                num_total_actors=args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
                shared_pg=pg if args.colocate_all_models else None,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_enable_sleep=args.vllm_enable_sleep,
                # args.vllm_num_engines,
                # args.vllm_tensor_parallel_size,
                # args.pretrain,
                # args.seed,
                # args.full_determinism,
                # args.enable_prefix_caching,
                # args.enforce_eager,
                # max_len,
                # args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size,
                # pg if args.colocate_all_models else None,
                # args.vllm_gpu_memory_utilization,
                # args.vllm_enable_sleep,
            )

    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
    )

    # if args.init_kl_coef == 0:
    #     ref_model = None
    # else:
    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
    )

    if not args.colocate_all_models:
        pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = PPORayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
        )
    else:
        critic_model = None

    # multiple reward models
    if not args.remote_rm_url:
        reward_pretrains = args.reward_pretrain.split(",")
        assert len(reward_pretrains) == 1, "Only one reward model is supported"
        reward_models = []
        for _ in reward_pretrains:
            reward_models.append(
                PPORayActorGroup(
                    args.reward_num_nodes,
                    args.reward_num_gpus_per_node,
                    RewardModelRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                )
            )
    else:
        reward_models = None

    # init reference/reward/actor model
    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    if not args.remote_rm_url:
        for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
            refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    ray.get(refs)

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # train actor and critic model
    refs = actor_model.async_fit_actor_model(
        critic_model, ref_model, reward_models, args.remote_rm_url, reward_fn=reward_fn, vllm_engines=vllm_engines
    )
    ray.get(refs)

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--evaluators", type=str, nargs='*', default=[], choices=['aime', 'math500', 'gpqa', 'aime_image'])
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_model",  action="store_true", default=False, help="Save HF model while saving checkpoint")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--ckpt_tag", type=str, default=None)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--actor_lr_schedule", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--kl_loss_coef", type=float, default=0, help="Coefficient for KL divergence penalty term in loss function")
    parser.add_argument("--kl_penalty_type", type=str, default="kl", help="KL penalty type")
    parser.add_argument("--kl_threshold_type", type=str, default=None, help="KL loss threshold type")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--train_vlm", action="store_true", default=False)
    parser.add_argument("--not_normalize_advantage", action="store_true", default=False, help="Not normalize advantage")


    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce++", "group_norm", "group_norm_pos", "group_norm_token_efficiency"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce++",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Agent
    parser.add_argument("--train_agent", action="store_true", default=False)
    parser.add_argument("--task_sampling_distributed", action="store_true", default=False)
    parser.add_argument("--task_group_distributed", action="store_true", default=False)
    parser.add_argument("--num_distributed_groups", type=int, default=1)
    parser.add_argument("--data_gather_redistribute", action="store_true", default=False)

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
    parser.add_argument("--env_reset_sleep_range", type=int, default=30)

    parser.add_argument("--agent_type", type=str, default='uitars')
    parser.add_argument("--agent_action_space", type=str, default='computer')
    parser.add_argument("--agent_prompt_language", type=str, default='Chinese')
    parser.add_argument("--num_history", type=int, default=None)
    parser.add_argument("--num_input_image", type=int, default=None)

    # LLM evaluation
    parser.add_argument("--use_llm_evaluator", action="store_true", default=False)
    parser.add_argument("--test_task_llm_eval", action="store_true", default=False)
    parser.add_argument("--api_type", type=str, default=None)
    parser.add_argument("--api_model", type=str, default=None)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--eval_prompt_file", type=str, default=None)
    parser.add_argument("--eval_prompt_dir", type=str, default=None)
    parser.add_argument("--llm_eval_temperature", type=float, default=0.0)
    parser.add_argument("--llm_eval_voting_type", type=str, default=None)
    parser.add_argument("--llm_eval_voting_num", type=int, default=1)

    # DAPO
    parser.add_argument("--use_dapo_trainer", action="store_true", default=False)
    parser.add_argument("--dapo_dynamic_sampling", action="store_true", default=False)
    parser.add_argument("--rollout_target_size", type=int, default=None)
    parser.add_argument("--max_num_gen_batches", type=int, default=None)
    parser.add_argument("--only_sample", action="store_true", default=False, help="can be used for statistic and eval")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--simple_load_dataset", action="store_true", default=False)
    parser.add_argument("--no_shuffle_train_dataset", action="store_true", default=False)
    
    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # MYQ UPDATE
    parser.add_argument("--apply_uncompleted_filter", action="store_true", default=False)
    parser.add_argument("--apply_query_filter", action="store_true", default=False)
    parser.add_argument("--apply_select_response_by_prm", action="store_true", default=False)
    parser.add_argument("--apply_select_response_longer_pos", action="store_true", default=False)
    parser.add_argument("--group_method", type=str, choices=['group_reward_incomplete_equal_to_neg', 'group_reward_with_learn_mask', 'normal'])
    parser.add_argument("--use_length_reward_in_efficiency", action="store_true", default=False)
    # End of UPDATE
    # ZBC UPDATE
    parser.add_argument("--random_temperature", action="store_true", default=False)
    # End of UPDATE

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        args.enable_prefix_caching = False
        print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache.")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template")
        # args.input_template = None
        # TODO: alse used to add custom format template

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False
        
    train(args)
