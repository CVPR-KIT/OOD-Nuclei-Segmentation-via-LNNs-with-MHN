import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.delta_var = delta_var  # Margin for variance loss
        self.delta_dist = delta_dist  # Margin for distance loss
        self.alpha = alpha  # Weight for variance term
        self.beta = beta  # Weight for distance term
        self.gamma = gamma  # Weight for regularization

    def forward(self, embeddings, instance_mask):
        """
        Args:
            embeddings: (B, C, H, W)  # C=4 (embedding dim)
            instance_mask: (B, H, W) with instance IDs (0=background)
        """
        batch_size, embed_dim, height, width = embeddings.shape
        losses = []

        for b in range(batch_size):
            emb = embeddings[b].permute(1, 2, 0).reshape(-1, embed_dim)  # (H*W, C)
            mask = instance_mask[b].flatten()  # (H*W,)

            # Ignore background (ID=0)
            valid = mask != 0
            if valid.sum() == 0:  # No instances in this sample
                continue

            emb = emb[valid]
            instance_ids = mask[valid]

            # --- Variance Loss (Pull same instances together) ---
            unique_ids = torch.unique(instance_ids)
            num_instances = len(unique_ids)
            means = []
            var_loss = 0

            for id in unique_ids:
                inst_emb = emb[instance_ids == id]
                mean_emb = inst_emb.mean(dim=0)
                means.append(mean_emb)
                var_loss += F.relu(torch.norm(inst_emb - mean_emb, dim=1).mean() - self.delta_var).pow(2).mean()

            var_loss /= num_instances

            # --- Distance Loss (Push different instances apart) ---
            dist_loss = 0
            if num_instances > 1:
                means = torch.stack(means)
                pairwise_dist = torch.norm(means[:, None] - means, dim=2)  # (N, N)
                eye = torch.eye(num_instances, device=emb.device)
                dist_loss = F.relu(self.delta_dist - pairwise_dist + eye * 1e6).pow(2).sum() / (num_instances * (num_instances - 1))

            # --- Regularization Loss ---
            reg_loss = torch.norm(emb, dim=1).mean()

            # Total loss for this sample
            total_loss = self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss
            losses.append(total_loss)

        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=embeddings.device)