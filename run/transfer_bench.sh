#!/usr/bin/bash --login

# Global settings
NUM_REPS=10
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
CONFIG_DIR=configs/transfer_bench
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    job_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb "
fi

launch () {
    dataset=$1

    run_script="python main.py --cfg ${CONFIG_DIR}/${dataset}-GPS+GPSE.yaml --repeat ${NUM_REPS} seed ${INIT_SEED} wandb.use ${USE_WANDB}"
    full_script="${job_script}${run_script}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

launch mnist
launch cifar
launch peptides-func
launch peptides-struct
