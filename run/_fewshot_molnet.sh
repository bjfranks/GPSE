#!/usr/bin/bash --login

# Global settings
NUM_REPS=$5
INIT_SEED=$6
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
CONFIGS=(
$4
)
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

for config in ${CONFIGS[@]}; do
    run_script="python main.py --cfg configs/molnet_bench/$1-GINE+${config}.yaml --repeat ${NUM_REPS} "
    run_script+="seed ${INIT_SEED} wandb.use ${USE_WANDB} train.record_individual_scores True"

    if [[ $WRAPPER != "local" ]]; then
        mkdir -p ${ROOT_DIR}/slurm_history
        run_script="sbatch -t 0-02:00:00 -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script}"
        run_script+=$2
    fi

    launch () {
        command=$1
        echo $command  # print out the command
        eval $command  # execute the command
    }

    launch "${run_script} name_tag $3train_size_128 dataset.subset_ratio 0.0078125"
    launch "${run_script} name_tag $3train_size_64 dataset.subset_ratio 0.015625"
    launch "${run_script} name_tag $3train_size_32 dataset.subset_ratio 0.03125"
    launch "${run_script} name_tag $3train_size_16 dataset.subset_ratio 0.0625"
    launch "${run_script} name_tag $3train_size_8 dataset.subset_ratio 0.125"
    launch "${run_script} name_tag $3train_size_4 dataset.subset_ratio 0.25"
    launch "${run_script} name_tag $3train_size_2 dataset.subset_ratio 0.5"
    launch "${run_script} name_tag $3train_size_1"
done
