#!/usr/bin/bash --login

# Global settings
NUM_REPS=1
INIT_SEED=1  # we can always extend the number of runs by keeping NUM_REPS=1 and then increment INIT_SEED
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
CONFIGs=(
AllPSE
GPSE
LapPE
none
rand
RWSE
)
USE_WANDB=False
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

for config in ${CONFIGS[@]}; do

  run_script="python main.py --cfg configs/mol_bench/zinc-GPS+${config},yaml --repeat ${NUM_REPS} "
  run_script+="seed ${INIT_SEED} wandb.use ${USE_WANDB} train.record_individual_scores True"

  if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    run_script="sbatch -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script} dataset.umg_split True"
  fi

  launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
  }

  launch "${run_script} name_tag train_size_64 dataset.umg_train_ratio 0.0125"
  launch "${run_script} name_tag train_size_32 dataset.umg_train_ratio 0.025"
  launch "${run_script} name_tag train_size_16 dataset.umg_train_ratio 0.05"
  launch "${run_script} name_tag train_size_8 dataset.umg_train_ratio 0.1"
  launch "${run_script} name_tag train_size_4 dataset.umg_train_ratio 0.2"
  launch "${run_script} name_tag train_size_2 dataset.umg_train_ratio 0.4"
  launch "${run_script} name_tag train_size_1"

done
