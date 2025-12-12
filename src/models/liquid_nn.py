import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

import matplotlib.pyplot as plt
import seaborn as sns


class LiquidNN(nn.Module):
    def __init__(
        self,
        feature_channels,
        hidden_size,
        embedding_size,
        ncp_type="CfC",
        exp_dir=None,
    ):
        super().__init__()

        self.feature_channels = feature_channels
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.sequence_length = len(feature_channels)

        self.feature_proj = nn.ModuleList([
            nn.Conv2d(ch, hidden_size, kernel_size=1) for ch in feature_channels
        ])

        self.wiring = AutoNCP(hidden_size, embedding_size)

        if ncp_type == "CfC":
            self.rnn = CfC(hidden_size, self.wiring, batch_first=True)
        elif ncp_type == "LTC":
            self.rnn = LTC(hidden_size, self.wiring, batch_first=True)
        else:
            raise ValueError("Unsupported ncp_type. Choose 'CfC' or 'LTC'")

        if exp_dir is not None:
            self.make_wiring_diagram(path=exp_dir, layout="kamada")

    def forward(self, features):
        """
        features: List[Tensor], each (B, C_i, H_i, W_i), expected order: [hd5, h4, h3, h2, h1]
        returns: embeddings (B, embedding_size, H0, W0) aligned to features[0]
        """
        assert isinstance(features, (list, tuple)) and len(features) > 0
        B0, _, H0, W0 = features[0].shape

        h_t = None
        last_out_2d = None

        for i, feat in enumerate(features):
            proj = F.relu(self.feature_proj[i](feat))  # (B, hidden_size, H, W)
            B, hidden_dim, H, W = proj.shape
            T = H * W

            proj_seq = proj.view(B, hidden_dim, T).permute(0, 2, 1).contiguous()  # (B, T, hidden_size)

            if h_t is None:
                h_t = torch.zeros((B, self.hidden_size), device=proj.device, dtype=proj.dtype)

            out, h_t = self.rnn(proj_seq, h_t)  # out: (B, T, embedding_size) (via wiring)

            B, T, emb_dim = out.shape
            out_2d = out.view(B, H, W, emb_dim).permute(0, 3, 1, 2).contiguous()  # (B, embedding_size, H, W)
            last_out_2d = out_2d

        embeddings = F.interpolate(last_out_2d, size=(H0, W0), mode="bilinear", align_corners=False)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def make_wiring_diagram(self, path: str, layout="kamada"):
        sns.set_style("white")
        plt.figure(figsize=(8, 8))
        legend_handles = self.wiring.draw_graph(layout=layout, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "LNN_wiring_diagram.png"))
        plt.close()
