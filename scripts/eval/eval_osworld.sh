CHECKPOINT=$1
ENV_URL=$2
ENV_MANAGER_PORT=$3
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

export PYTHONPATH=./:$PYTHONPATH 

python -m openrlhf.cli.eval_agent \
    --env_type osworld \
    --env_url $ENV_URL \
    --env_manager_port $ENV_MANAGER_PORT \
    --pretrain $CHECKPOINT \
    --agent_type uitars \
    --agent_action_space computer \
    --vllm_tensor_parallel_size $TENSOR_PARALLEL \
    --save_trajectory
