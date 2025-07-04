##############################
# slurm settings
##############################
NNODES=${NNODES:-1}
GPUS_PER_TASK=8 # GPU per node
CPUS_PER_TASK=96 # CPU per node
##############################

export GPUS=$((GPU_PER_NODE * NNODES))

srun --ntasks ${NNODES} \
    --ntasks-per-node 1 \
    --gpus-per-task ${GPUS_PER_TASK} \
    --cpus-per-task ${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SLURM_ARGS} \
    bash $(dirname $0)/ray_launch.sh
