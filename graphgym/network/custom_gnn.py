import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgym.layer.gatedgcn_layer import GatedGCNLayer, ResGatedGCNConvLayer
from graphgym.layer.gine_conv_layer import GINEConvLayer, GINConvLayer
from graphgym.layer.gatv2conv import GATv2ConvLayer


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        print(register.head_dict.keys())
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        if model_type == 'gatedgcnconv_noef':
            return ResGatedGCNConvLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'ginconv':
            return GINConvLayer
        elif model_type == 'gatv2conv':
            return GATv2ConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
