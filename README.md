# Towards Graph Foundation Models: A Study on the Generalization of Positional and Structural Encodings

## Installation

This codebase is built on top of
[GPSE](https://github.com/G-Taxonomy-Workgroup/GPSE) which is built on
[GraphGym](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html)
and [GraphGPS](https://github.com/rampasek/GraphGPS). Follow the steps below to
set up dependencies, such as [PyTorch](https://pytorch.org/) and
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/):

```bash
# Create a conda environment for this project
conda create -n gpse python=3.10 -y && conda activate gpse

# Install main dependencies PyTorch and PyG
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg=2.2 -c pyg -c conda-forge -y
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge -y

# Install the rest of the pinned dependencies
pip install -r requirements.txt

# Clean up cache
conda clean --all -y
```

## Quick start

You can pre-train the GPSE model from scratch using the configs provided, e.g.

```bash
python main.py --cfg configs/pretrain/gpse_molpcba.yaml
```

```bash
python main.py --cfg configs/pretrain/gpse+_molpcba.yaml
```

```bash
python main.py --cfg configs/pretrain/gpse-_molpcba.yaml
```

After the pre-training is done, you need to manually move the checkpointed model to the `pretrained_models/` directory.
The checkpoint can be found under `results/gpse_molpcba/<seed>/ckpt/<best_epoch>.pt`, where `<seed>` is the random seed
for this run (0 by default), and `<best_epoch>` is the best epoch number (you will only have one file, that *is* the
best epoch).

### Run downstream evaluations

After you have prepared the pre-trained model `gpse_molpcba.pt`, you can then run downstream evaluation for models that
uses `GPSE` encoded features. For example, to run the `ZINC` benchmark:

```bash
python main.py --cfg configs/mol_bench/zinc-GPS+GPSE.yaml
```

You can also execute batch of runs using the run scripts prepared under `run/`. For example, to run all the data-scarce experiments for ZINC-12k (except for GPSE+ and GPSE-)

```bash
sh run/fewshot_zinc.sh
```

And to run the experiment for GPSE+ and GPSE- with seed 4:

```bash
sh run/_fewshot_zinc.sh 4 " posenc_GPSE.model_dir pretrained_models/gpse+_molpcba.pt" GPSE+
```

```bash
sh run/_fewshot_zinc.sh 4 " posenc_GPSE.model_dir pretrained_models/gpse-_molpcba.pt posenc_GPSE.rand_type FixedSE" GPSE-
```

## Known issues

- Incompatibility with PyG 2.4.0 due to a minor bug in the GraphGym MLP construction (see https://github.com/G-Taxonomy-Workgroup/GPSE/issues/1 and https://github.com/pyg-team/pytorch_geometric/issues/8484).
