import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import GATConv


class FingerprintMLP(nn.Module):
    def __init__(self, fp_dim=167, hidden_feats=32, drop_prob=0.2, activation="relu"):
        super(FingerprintMLP, self).__init__()
        self.fc1 = nn.Linear(fp_dim, hidden_feats)
        self.dropout = nn.Dropout(drop_prob)
        if activation == "relu":
            self.activation = nn.ReLU()
        self._init_weights()

    def forward(self, fp):
        fp = self.fc1(fp)
        fp = self.activation(fp)
        fp = self.dropout(fp)
        return fp

    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)


class GraphAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=3, heads=4, dropout=0.2):
        super(GraphAttention, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)])

        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return x


def load_mpnn_to_gflow(saved_model_path):
    model = CombinedRepresentation()
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.eval()
    return model


class CombinedRepresentation(nn.Module):
    def __init__(self, fp_dim=167, graph_in_channels=32, hidden_dim=32, output_dim=1):
        super(CombinedRepresentation, self).__init__()
        self.fp_net = FingerprintMLP(fp_dim=fp_dim, hidden_feats=hidden_dim)
        self.gat_net = GraphAttention(in_channels=graph_in_channels, hidden_channels=hidden_dim)

        # Combined dimension will be hidden_dim * 2 (concatenated)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, fp, x, edge_index, batch):
        fp_repr = self.fp_net(fp)
        graph_repr = self.gat_net(x, edge_index, batch)

        # Concatenate the two representations
        combined = torch.cat([fp_repr, graph_repr], dim=1)
        # Pass through final MLP
        output = self.final_mlp(combined)
        return output
