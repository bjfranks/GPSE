import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_scatter import scatter


@register_layer('gatedgraphconv')
class GatedGraphConvGraphGymLayer(nn.Module):
    """Edge attr aware GAT convolution layer.
    """

    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.model = GatedGraphConv(
            layer_config.dim_out,
            num_layers=cfg.gnn.layers_mp,
            aggr='add',
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        x = self.model(batch.x, batch.edge_index, batch.edge_weight)
        batch.x = x
        return batch