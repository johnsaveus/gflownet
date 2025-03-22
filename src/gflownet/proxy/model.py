import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm.graph_norm import GraphNorm


class GraphAttention(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_dim,
        gnn_layers,
        gnn_channels,
        heads,
        dropout_proba,
        gnn_norm=False,
        mlp_layers=1,
        mlp_channels=64,
        mlp_norm=False,
    ):
        super(GraphAttention, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_channels=node_feats,
                out_channels=gnn_channels,
                heads=heads,
                dropout=dropout_proba,
                concat=True,
                edge_dim=edge_dim,
            )
        )
        for l in range(gnn_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=gnn_channels * heads,
                    out_channels=gnn_channels,
                    heads=heads,
                    dropout=dropout_proba,
                    concat=True,
                    edge_dim=edge_dim,
                )
            )
        self.gat_layers.append(
            GATConv(
                in_channels=gnn_channels * heads,
                out_channels=gnn_channels,
                heads=heads,
                dropout=dropout_proba,
                concat=False,
                edge_dim=edge_dim,
            )
        )
        self.dropout = nn.Dropout(dropout_proba)
        self.activation = nn.LeakyReLU()
        self.gnn_norm = gnn_norm
        if gnn_norm:
            self.graph_norms = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for l in range(gnn_layers):
                out_dim = gnn_channels * heads if l < gnn_layers - 1 else gnn_channels
                self.graph_norms.append(GraphNorm(out_dim))
                self.batch_norms.append(nn.BatchNorm1d(out_dim))
        self.mlp = FullyConnected(
            gnn_channels,
            mlp_layers,
            mlp_channels,
            dropout_proba,
            mlp_norm,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for l, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            if self.gnn_norm:
                x = self.graph_norms[l](x, batch)
                x = self.batch_norms[l](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, input_dim, layers, channels, dropout_proba, norm=False):
        super(FullyConnected, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_dim, channels))
        for _ in range(layers - 2):
            self.fc_layers.append(nn.Linear(channels, channels))
        self.fc_layers.append(nn.Linear(channels, 1))
        self.dropout = nn.Dropout(dropout_proba)
        self.activation = nn.LeakyReLU()
        self.norm = norm
        if norm:
            self.batch_norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if not i == len(self.fc_layers) - 1:
                if self.norm:
                    x = self.batch_norm(x)
                x = self.activation(x)
                x = self.dropout(x)
        return x
