import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from .discrim_loss import DiscriminativeLoss
from .lovasz_losses import lovasz_hinge

class NucleiInstanceLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5, 
                 lambda_quant=0.1, lambda_seed=0.5, lambda_smooth=0.2):
        super().__init__()
        # Discriminative Loss Components
        #self.discriminative_loss = DiscriminativeLoss(delta_var=delta_var, delta_dist=delta_dist)

        self.quant_flag = False
        
        # Lovasz-Sigmoid for Mask IoU
        self.lovasz_loss = LovaszSigmoidLoss()
        
        # Seed Loss for seed map
        self.seed_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.lambda_quant = lambda_quant
        self.lambda_seed = lambda_seed
        self.lambda_smooth = lambda_smooth

    def forward(self, outputs, gt_masks, quant_loss_image, quant_loss_lnn):
        """
        Args:
            outputs: (B,4,H,W) - [semantic_mask, seed_map, emb1, emb2]
            gt_masks: (B,H,W) instance IDs
            quant_loss_image: Tuple of 8 quantization losses from image encoder
            quant_loss_lnn: Tuple of 8 quantization losses from liquid NN
        """
        self.quant_flag = False if quant_loss_image is None and quant_loss_lnn is None else True
        #print(f"Quant Flag: {self.quant_flag}")
        


        # Split outputs into components
        semantic_mask = outputs[:, 0]  # For Lovasz (B,H,W)
        seed_logits = outputs[:, 1]    # For seed loss (B,H,W)
        embeddings = outputs[:, 2:]    # For discriminative loss (B,2,H,W)

        # --- Quantization Loss ---
        # average quant losses
        if self.quant_flag:
            quantloss_image = sum([t.mean() for t in quant_loss_image]) / len(quant_loss_image)
            #print(f"Quant Loss_i: {quantloss_image}")
            quantloss_lnn = sum([t.mean() for t in quant_loss_lnn]) / len(quant_loss_lnn)
            #print(f"Quant Loss_l: {quantloss_lnn}")
            quant_loss = quantloss_image + quantloss_lnn 
            #print(f"Quant Loss: {quant_loss}")
        else:
            quant_loss = torch.tensor(0.0).to(semantic_mask.device)
        
        
        
        # --- Discriminative Loss ---
        #disc_loss = self.discriminative_loss(embeddings, gt_masks)
        #print(f"Disc Loss: {disc_loss}")
        
        # --- Lovasz-Sigmoid Loss ---
        lovasz_loss = self.lovasz_loss(semantic_mask, (gt_masks > 0).float())
        #print(f"Lovasz Loss: {lovasz_loss}")
        
        # --- Seed Loss ---
        seed_gt = self._generate_seed_gt(gt_masks).to(seed_logits.device)
        seed_loss = self.seed_loss(seed_logits, seed_gt)
        #print(f"Seed Loss: {seed_loss}")
        
        # --- Smoothness Loss ---
        smooth_loss = self._smoothness_loss(embeddings)
        #print(f"Smooth Loss: {smooth_loss}")
        
        # Total Loss
        total_loss = (
            lovasz_loss +
            self.lambda_quant * quant_loss +
            #disc_loss +
            self.lambda_seed * seed_loss +
            self.lambda_smooth * smooth_loss
        )
        
        return total_loss, (quant_loss, lovasz_loss, seed_loss, smooth_loss)

    def _generate_seed_gt(self, instance_mask):
        # Create center heatmap using instance centroids
        batch_size = instance_mask.size(0)
        heatmaps = []
        for b in range(batch_size):
            mask = instance_mask[b].cpu().numpy()
            heatmap = np.zeros_like(mask, dtype=np.float32)
            
            for inst_id in np.unique(mask):
                if inst_id == 0: continue
                y, x = np.where(mask == inst_id)
                center_x, center_y = x.mean(), y.mean()
                heatmap = cv2.circle(heatmap, 
                                   (int(center_x), int(center_y)),
                                   radius=2, color=1, thickness=-1)
            heatmaps.append(torch.from_numpy(heatmap))
        
        return torch.stack(heatmaps).to(instance_mask.device)

    def _smoothness_loss(self, embeddings):
        # Spatial consistency loss
        dx = torch.abs(embeddings[..., :-1] - embeddings[..., 1:])
        dy = torch.abs(embeddings[..., :-1, :] - embeddings[..., 1:, :])
        return dx.mean() + dy.mean()

class LovaszSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Ensure pred is in [0,1] via sigmoid
        pred = torch.sigmoid(pred)
        return lovasz_hinge(pred, target, per_image=True)