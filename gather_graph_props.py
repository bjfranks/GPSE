import json
import pickle
import sys

import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import set_cfg, load_cfg


from networkx.algorithms.approximation import node_connectivity, large_clique_size
from networkx.algorithms.cluster import clustering, triangles
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.centrality import degree_centrality
from networkx.classes.function import density, degree_histogram

sys.path.append(".")
sys.path.append("..")

from graphgym.loader.master_loader import *


def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError("crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)

        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('ogbn'):
            dataset = preformat_OGB_Node(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)

            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))

            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")

    elif format == 'OpenMolGraph':
        dataset = preformat_OpenMolGraph(dataset_dir, name=name)

    elif format == 'SyntheticWL':
        dataset = preformat_SyntheticWL(dataset_dir, name=name)

    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Preprocess for reducing the molecular dataset to unique structured graphs
    cfg.dataset.unique_mol_graphs = False
    if cfg.dataset.unique_mol_graphs:
        dataset = get_unique_mol_graphs_via_smiles(dataset,
                                                   cfg.dataset.umg_train_ratio,
                                                   cfg.dataset.umg_val_ratio,
                                                   cfg.dataset.umg_test_ratio,
                                                   cfg.dataset.umg_random_seed)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith(('posenc_', 'graphenc_')) and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pretrained_gnn, _recover_orig_cfgs = load_pretrained_gnn(cfg)
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg,
                                        pretrained_gnn=pretrained_gnn),
                                show_progress=True)
        _recover_orig_cfgs()  # HACK: recover original global configs
        if hasattr(dataset.data, "y") and len(dataset.data.y.shape) == 2:
            cfg.share.num_node_targets = dataset.data.y.shape[1]
        if hasattr(dataset.data, "y_graph"):
            cfg.share.num_graph_targets = dataset.data.y_graph.shape[1]
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    if cfg.graph_norm.enable:
        pre_transform_in_memory(dataset, GraphNormalizer(), show_progress=True)

    dataset.transform_list = None
    randse_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('randenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            randse_enabled_list.append(pe_name)
    if randse_enabled_list:
        set_random_se(dataset, randse_enabled_list)

    if cfg.virtual_node:
        set_virtual_node(dataset)

    if dataset.transform_list is not None:
        dataset.transform = T.Compose(dataset.transform_list)

    # # Set standard dataset train/val/test splits
    # if hasattr(dataset, 'split_idxs'):
    #     set_dataset_splits(dataset, dataset.split_idxs)
    #     delattr(dataset, 'split_idxs')
    #
    # # Verify or generate dataset train/val/test splits
    # prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    logging.info(f"Finished processing data:\n  {dataset.data}")

    return dataset

def avg_num_nodes(g_list):
    return np.mean([G.number_of_nodes() for G in g_list])


def avg_num_edges(g_list):
    return np.mean([G.number_of_edges() for G in g_list])


def avg_density(g_list):
    return np.mean([density(G) for G in g_list])


def node_conn(g_list):
    return np.mean([node_connectivity(G) for G in g_list])


def avg_degree(g_list):
    deg_list = list()
    for G in g_list:
        G_deg = degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        deg_list.append(sum(G_deg_sum) / G.number_of_nodes())
    return np.mean(deg_list)


def num_triangles(g_list):
    return np.mean([sum(triangles(G).values())/3 for G in g_list])


def diam(g_list):
    def compute(G):
        try:
            return diameter(G)
        except:
            return 1

    return np.mean([compute(G) for G in g_list])


def max_cli(g_list):
    return np.mean([large_clique_size(G) for G in g_list])


def deg_cent(g_list):
    return np.mean([np.mean(list(degree_centrality(G).values())) for G in g_list])


def avg_cluster_coeff(g_list):
    return np.mean([np.mean(list(clustering(G).values())) for G in g_list])


datasets = [
            ('PyG-ZINC', 'subset'),
            ('PyG-GNNBenchmarkDataset', 'CIFAR10'),
            ('PyG-GNNBenchmarkDataset', 'MNIST'),
            ('PyG-GNNBenchmarkDataset', 'CSL'),
            ('SyntheticWL', 'EXP'),
            ('OGB', 'ogbg-molhiv'),
            ('OGB', 'ogbg-molpcba'),
            ('OGB', 'ogbg-molbbbp'),
            ('OGB', 'ogbg-molbace'),
            ('OGB', 'ogbg-moltox21'),
            ('OGB', 'ogbg-moltoxcast'),
            ('OGB', 'ogbg-molsider'),
            ('OGB', 'PCQM4Mv2-subset'),
            ('OGB', 'peptides-functional'),
            ('OGB', 'peptides-structural'),
            ('OGB', 'ogbn-arxiv'),
            ('OGB', 'ogbn-proteins'),
            ]

algos = {'num_nodes': avg_num_nodes,
         'num_edges': avg_num_edges,
         'density': avg_density,
         'connectivity': node_conn,
         'diameter': diam,
         'max_clique': max_cli,
         'centrality': deg_cent,
         'cluster_coeff': avg_cluster_coeff,
         'triangles': num_triangles
         }


def compute_metrics(dataset, algos):
    x = [to_networkx(d, to_undirected=True) for d in dataset]
    # y = dataset.data.y.numpy()

    metrics = dict()
    for name, fn in algos.items():
        metrics[name] = fn(x)

    return metrics


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    cfg.train.mode = None  # XXX: temporary fix, need to register train.mode
    load_cfg(cfg, args)

    dataset_dict = dict()
    for format, name in datasets:
        dataset_dict[f'{format}_{name}'] = dict()
        dataset = load_dataset_master(format, name, './datasets')
        metrics = compute_metrics(dataset, algos)
        for k in metrics.keys():
            dataset_dict[f'{format}_{name}'][k] = metrics[k]
        print(f'{name} done!')

        with open('graph_props.pkl', 'wb') as fp:
            pickle.dump(dataset_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('graph_props.json', 'w') as fp:
            json.dump(dataset_dict, fp, indent=4)
