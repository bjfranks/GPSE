#!/usr/bin/bash --login

# Global settings
DATASETS=(
bace
bbbp
clintox
hiv
muv
sider
tox21
toxcast
)
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
#################

HOME_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname $HOME_DIR)

cd $ROOT_DIR

for dataset in ${DATASETS[@]}; do
    for seed in ${SEEDS[@]}; do
        sh run/_fewshot_molnet.sh ${dataset} "$1" "$2" 1 ${seed}
    done
done
