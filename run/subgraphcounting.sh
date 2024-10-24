#!/usr/bin/bash --login

# Global settings
WRAPPER=wrapper_rptu  # local, wrapper_msuicer, wrapper_mila, wrapper_rptu
SEEDS=(
1
2
3
4
5
6
7
8
9
10
)
EXPERIMENTS=(
0
1
2
3
)
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)
echo HOME_DIR=$HOME_DIR
echo ROOT_DIR=$ROOT_DIR

cd $ROOT_DIR

run_script="python main.py --cfg configs/wl_bench/CC-GIN+GPSE.yaml "

if [[ $WRAPPER != "local" ]]; then
    mkdir -p ${ROOT_DIR}/slurm_history
    run_script="sbatch -t 0-02:00:00 -c 5 --mem=45GB -o ${ROOT_DIR}/slurm_history/slurm-%A.out run/${WRAPPER}.sb ${run_script}"
fi

launch () {
    command=$1
    echo $command  # print out the command
    eval $command  # execute the command
}

for exp in ${EXPERIMENTS[@]}; do
    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cc${exp} name_tag GPSE#${exp} seed ${seed}"
    done

    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cc${exp} posenc_GPSE.model_dir pretrained_models/gpse+_molpcba.pt name_tag GPSE+#${exp} seed ${seed}"
    done

    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cc${exp} posenc_GPSE.model_dir pretrained_models/gpse-_molpcba.pt posenc_GPSE.rand_type FixedSE name_tag GPSE-#${exp} seed ${seed}"
    done
done

EXPERIMENTS=(
0
1
2
3
4
)
for exp in ${EXPERIMENTS[@]}; do
    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cg${exp} name_tag GPSE_${exp} seed ${seed}"
    done

    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cg${exp} posenc_GPSE.model_dir pretrained_models/gpse+_molpcba.pt name_tag GPSE+_${exp} seed ${seed}"
    done

    for seed in ${SEEDS[@]}; do
        launch "${run_script} dataset.name cg${exp} posenc_GPSE.model_dir pretrained_models/gpse-_molpcba.pt posenc_GPSE.rand_type FixedSE name_tag GPSE-_${exp} seed ${seed}"
    done
done
