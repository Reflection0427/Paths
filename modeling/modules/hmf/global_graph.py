import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HMFGlobalGraphFusion(nn.Module):
    """
    HMF global branch: dynamic cross-modal graph construction and aggregation.
    """

    def __init__(self, dim, num_heads=8, k_intra=3, k_cross=3, temperature=0.05):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.k_intra = k_intra
        self.k_cross = k_cross
        self.temperature = temperature

        self.W_Q_select = nn.Linear(dim, dim)
        self.W_K_select = nn.Linear(dim, dim)

        self.sigma = nn.Parameter(torch.tensor(2.0))
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)

        self.W_node = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def get_temporal_bias(self, T, device):
        indices = torch.arange(T, device=device).float()
        dist_matrix = (indices.unsqueeze(0) - indices.unsqueeze(1)).pow(2)
        bias = torch.exp(-dist_matrix / (2 * self.sigma.pow(2) + 1e-5))
        return bias.unsqueeze(0)

    def forward(self, x_query, x_intra_pool, x_cross_pool):
        B, T, D = x_query.shape
        H = self.num_heads
        d_k = self.head_dim

        temporal_bias = self.get_temporal_bias(T, x_query.device) * self.bias_scale

        query_select = self.W_Q_select(x_query)
        key_intra_select = self.W_K_select(x_intra_pool)
        key_cross_select = self.W_K_select(x_cross_pool)

        k_intra = min(self.k_intra, x_intra_pool.shape[1])
        k_cross = min(self.k_cross, x_cross_pool.shape[1])

        score_intra = torch.bmm(query_select, key_intra_select.transpose(1, 2)) / self.temperature
        score_intra = score_intra + temporal_bias
        _, idx_intra = torch.topk(score_intra, k=k_intra, dim=-1)

        score_cross = torch.bmm(query_select, key_cross_select.transpose(1, 2)) / self.temperature
        score_cross = score_cross + temporal_bias
        _, idx_cross = torch.topk(score_cross, k=k_cross, dim=-1)

        def gather_neighbors(pool, idx):
            pool_expand = pool.unsqueeze(1).expand(-1, T, -1, -1)
            idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, D)
            return torch.gather(pool_expand, 2, idx_expand)

        neighbors_intra = gather_neighbors(x_intra_pool, idx_intra)
        neighbors_cross = gather_neighbors(x_cross_pool, idx_cross)
        center_node = x_query.unsqueeze(2)
        all_neighbors = torch.cat([center_node, neighbors_intra, neighbors_cross], dim=2)
        num_neighbors = all_neighbors.shape[2]

        q_heads = self.W_q(x_query).view(B, T, H, d_k)
        k_heads = self.W_k(all_neighbors).view(B, T, num_neighbors, H, d_k).permute(0, 1, 3, 2, 4)
        v_heads = self.W_v(all_neighbors).view(B, T, num_neighbors, H, d_k).permute(0, 1, 3, 2, 4)

        attn_scores = torch.matmul(q_heads.unsqueeze(3), k_heads.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        agg_heads = torch.matmul(attn_weights, v_heads).squeeze(3)

        agg_features = agg_heads.reshape(B, T, D)
        hyperedge_feature = self.W_out(agg_features)

        gate_input = torch.cat([hyperedge_feature, x_query], dim=-1)
        beta = torch.sigmoid(self.W_node(gate_input))
        return self.layer_norm(x_query + beta * hyperedge_feature)
