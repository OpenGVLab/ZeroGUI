SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

#############################
JOBID=$1
NODELIST=$2
NNODES=${NNODES:-4}
GPUS_PER_TASK=8
CPUS_PER_TASK=96
SCRIPT=$SCRIPT_DIR/train.sh
#############################

export GPUS=$((GPU_PER_NODE * NNODES))

srun --jobid=${JOBID} \
  --nodelist=${NODELIST} \
  --ntasks ${NNODES} \
  --ntasks-per-node 1 \
  --gpus-per-task ${GPUS_PER_TASK} \
  --cpus-per-task ${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  bash $SCRIPT