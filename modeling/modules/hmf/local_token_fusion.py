import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HMFLocalTokenFusionHead(nn.Module):
    """
    HMF local branch: token alignment, local fusion, and attention pooling.
    """

    def __init__(self, in_dim=768):
        super().__init__()
        self.in_dim = in_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, in_dim))
        self.scale = in_dim ** -0.5
        self.attn_norm = nn.LayerNorm(in_dim)

    def forward(self, x_rgb, x_evt, return_tokens=False):
        B, _, D = x_rgb.shape
        device = x_rgb.device

        rgb_norm = F.normalize(x_rgb, dim=-1)
        evt_norm = F.normalize(x_evt, dim=-1)
        sim_matrix = torch.bmm(rgb_norm, evt_norm.transpose(1, 2))
        cost_matrix = 1 - sim_matrix

        cost_cpu = cost_matrix.detach().float().cpu().numpy()
        aligned_indices = []
        with torch.no_grad():
            for b in range(B):
                _, col_ind = linear_sum_assignment(cost_cpu[b])
                aligned_indices.append(torch.from_numpy(col_ind))

        col_indices = torch.stack(aligned_indices).to(device)
        gather_indices = col_indices.unsqueeze(-1).expand(-1, -1, D)
        x_evt_aligned = torch.gather(x_evt, 1, gather_indices)

        loss_align = (1 - F.cosine_similarity(x_rgb, x_evt_aligned, dim=-1)).mean()

        cat_feat = torch.cat([x_rgb, x_evt_aligned], dim=-1)
        gate = self.fusion_gate(cat_feat)
        x_fused = gate * x_rgb + (1 - gate) * x_evt_aligned
        x_fused = self.attn_norm(x_fused)

        query = self.pool_query.expand(B, -1, -1)
        attn_logits = torch.bmm(query, x_fused.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        global_feat = torch.bmm(attn_weights, x_fused).squeeze(1)

        if return_tokens:
            return global_feat, loss_align, x_rgb, x_evt_aligned, x_fused
        return global_feat, loss_align
