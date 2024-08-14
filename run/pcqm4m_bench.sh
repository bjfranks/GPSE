#!/usr/bin/bash --login

if [ $# -eq 0 ]; then
    echo "No seed provided. Usage: $0 <seed>"
    exit 1
fi

# Global settings
INIT_SEED=$1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
CONFIG_DIR=configs/mol_bench
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    job_script="sbatch -c 5 --mem=45GB -t 2-00:00:00 -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb "
fi

launch () {
    dataset=$1
    model=$2
    pse=$3

    name="${dataset}-${model}+${pse}"
    run_script="python main.py --cfg ${CONFIG_DIR}/${name}.yaml --repeat 1 seed ${INIT_SEED} wandb.use ${USE_WANDB}"
    full_script="${job_script}${run_script}"

    echo $full_script  # print out the command
    eval $full_script  # execute the command
}

# ZINC

# PCQM4Mv2-subset
launch pcqm4msubset GPS none
launch pcqm4msubset GPS rand
launch pcqm4msubset GPS LapPE
launch pcqm4msubset GPS RWSE
launch pcqm4msubset GPS GPSE
