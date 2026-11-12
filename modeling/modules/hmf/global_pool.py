import torch.nn as nn
import torch


class HMFGlobalConsensusPooling(nn.Module):
    """
    HMF global branch: cross-modal temporal consensus pooling.
    """

    def __init__(self, dim):
        super().__init__()
        self.consensus_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_rgb, x_evt):
        joint_feat = torch.cat([x_rgb, x_evt], dim=-1)

        scores = self.consensus_net(joint_feat)
        score_rgb = scores[:, :, 0].unsqueeze(-1)
        score_evt = scores[:, :, 1].unsqueeze(-1)

        weight_rgb = self.softmax(score_rgb)
        weight_evt = self.softmax(score_evt)

        rgb_pool = (x_rgb * weight_rgb).sum(dim=1)
        evt_pool = (x_evt * weight_evt).sum(dim=1)
        return rgb_pool, evt_pool
