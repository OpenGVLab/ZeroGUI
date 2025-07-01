set -ex

ROOT=$PWD
cd $(dirname $0)

WORLD_SIZE=$((SLURM_NTASKS))
RANK=$((SLURM_PROCID))
MASTER_ADDR=$(scontrol show hostname ${SLURM_STEP_NODELIST} | head -n1)
MASTER_ADDR=$(echo $MASTER_ADDR | cut -d '-' -f 3-6 | tr '-' '.')
MASTER_PORT=29500
echo $MASTER_ADDR 
echo $MASTER_PORT 
echo $WORLD_SIZE 
echo $RANK
export DASHBORAD_PORT=10000

export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=$ROOT:$PYTHONPATH
# export HF_HOME=$(realpath $PWD/../../cache/huggingface/)

num_nodes=$WORLD_SIZE
start_time=$(date +%Y%m%d%H%M)

# num_nodes has to be at least 1
if [ $num_nodes -lt 1 ]; then
    echo "Number of nodes must be at least 1"
    exit 1
fi

# if HOST contains "master", then this is the head node
if [[ $RANK -eq 0 ]]; then
    node_role="master"
else
    node_role="worker"
fi
head_node_ip=$MASTER_ADDR

script="grpo.sh"

# logging
N_SAMPLES=${N_SAMPLE:-64}
TBS=${TBS:-2048}
RBS=${RBS:-1}
R_TARGET_SIZE=${R_TARGET_SIZE:-2048}
KL=${KL:-1e-1}
LR=${LR:-2e-6}
TEMP=${TEMP:-0.5}
EXP_FLAG=${EXP_FLAG:-""}
POLICY_TYPE=${POLICY_TYPE:-"ppo"}
KL_TYPE=${KL_TYPE:-"low_var_kl"}
LOG_DIR=log/${EXP_FLAG}-kl_${KL_TYPE}_${KL}-rbs_${RBS}-sample_${N_SAMPLES}-rtarget_${R_TARGET_SIZE}-tbs_${TBS}-lr_${LR}-temp_${TEMP}
mkdir -p $(dirname $0)/$LOG_DIR

wait_time=15
if [ "$node_role" == "master" ]; then
    echo "Starting Ray head node..."
    # Start Ray on this node as the head node and extract its address
    # The `ray start --head` command outputs information that includes the address,
    # but here we're assuming it's known or statically assigned for simplicity.
    ray start --head --dashboard-host 0.0.0.0 --port=6379 --ray-debugger-external --dashboard-port=$DASHBORAD_PORT --resources '{"COMPUTE": 100000000000000.0, "HEAD": 100000000000000.0}'
    sleep $wait_time
elif [ "$node_role" == "worker" ]; then
    sleep $wait_time
    attempt=1
    echo "Starting Ray worker node and attempting to connect to the head node at $head_node_ip:6379"
    while true; do
        # Attempt to start Ray and connect to the head node
        ray start --address="$head_node_ip:6379" --dashboard-port=$DASHBORAD_PORT --resources '{"COMPUTE": 100000000000000.0, "virtual_cluster_default": 100000000000000.0}'  && break || {
            if [ $attempt -le 5 ]; then
                echo "Ray worker start attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        }
    done
fi
# run the training script once Ray has been started on all nodes
sleep $wait_time
if [ "$node_role" == "master" ]; then
    num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
    echo "Number of active Ray nodes: $num_active_ray_nodes"
    if [ $num_active_ray_nodes -lt $num_nodes ]; then
        echo "Waiting for all Ray nodes to start..."
        attempt=1
        while true; do
            num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
            if [ $num_active_ray_nodes -eq $num_nodes ]; then
                break
            elif [ $attempt -le 5 ]; then
                echo "python command attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        done
    fi
    echo "End starting"
    # python examples/scripts/test_ray.py
    sh $(dirname $0)/${script} $RANK 2>&1 | tee $LOG_DIR/grpo_ray_${num_nodes}_${node_role}_${RANK}.log
else
    echo "End starting"
    sh $(dirname $0)/${script} $RANK 2>&1 | tee $LOG_DIR/grpo_ray_${num_nodes}_${node_role}_${RANK}.log
    sleep infinity
fi
