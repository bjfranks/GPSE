#!/usr/bin/bash --login

# Global settings
NUM_REPS=1
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
CONFIG=configs/pretrain/gpse_molpcba_ablation_template.yaml
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

run_script+="python main.py --cfg ${CONFIG} --repeat ${NUM_REPS} "
run_script+="seed ${INIT_SEED} wandb.use ${USE_WANDB} train.record_individual_scores True"

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    run_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script}"
fi

launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
}

launch "${run_script} name_tag train_size_64 dataset.umg_train_ratio 0.015625"
launch "${run_script} name_tag train_size_32 dataset.umg_train_ratio 0.03125"
launch "${run_script} name_tag train_size_16 dataset.umg_train_ratio 0.0625"
launch "${run_script} name_tag train_size_8 dataset.umg_train_ratio 0.125"
launch "${run_script} name_tag train_size_4 dataset.umg_train_ratio 0.25"
launch "${run_script} name_tag train_size_2 dataset.umg_train_ratio 0.5"
