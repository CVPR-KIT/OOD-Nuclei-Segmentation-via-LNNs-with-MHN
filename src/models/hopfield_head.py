import torch
import torch.nn as nn
import torch.nn.functional as F
from hflayers import HopfieldLayer 

class HopfieldHead(nn.Module):
    def __init__(
        self,
        use_lnn=True,
        image_channels=512,
        lnn_channels=16,
        shared_dim=256,
        hopfield_num_heads=4,
        hopfield_scaling=1.0,
        hopfield_update_steps_max=3,
        hopfield_update_steps_eps=1e-4,
        hopfield_num_patterns=256,  # Number of stored patterns
        hopfield_association_activation='relu',
        hopfield_dropout=0.2,
        hopfield_batch_first=True,
        hopfield_trainable=True,
    ):
        super(HopfieldHead, self).__init__()
        self.use_lnn = use_lnn
        
        # Projection layers
        # Always project image channels
        self.image_proj = nn.Conv2d(image_channels, shared_dim, kernel_size=1)

        # Only define LNN-related layers if using LNN
        if self.use_lnn:
            self.lnn_proj = nn.Conv2d(lnn_channels, shared_dim, kernel_size=1)
            # Combine projected image and downsampled LNN into shared_dim
            self.combine_proj = nn.Conv2d(image_channels + lnn_channels, shared_dim, kernel_size=1)
        
        # Initialize Hopfield layer
        self.hopfield = HopfieldLayer(
            input_size=shared_dim,            # Depth of the input (state pattern)
            hidden_size=shared_dim,           # Depth of the association space
            output_size=image_channels,       # Depth of the output projection (512)
            pattern_size=shared_dim,          # Depth of patterns to be selected
            num_heads=hopfield_num_heads,     # Number of parallel association heads
            scaling=hopfield_scaling,         # Scaling factor
            update_steps_max=hopfield_update_steps_max,
            update_steps_eps=hopfield_update_steps_eps,
            lookup_weights_as_separated=False,
            lookup_targets_as_trainable=hopfield_trainable,
            stored_pattern_size=shared_dim,
            pattern_projection_size=shared_dim,
            batch_first=hopfield_batch_first,
            association_activation=hopfield_association_activation,
            dropout=hopfield_dropout,
            input_bias=True,
            concat_bias_pattern=False,
            add_zero_association=False,
            disable_out_projection=False,
            quantity=hopfield_num_patterns,    # Number of stored patterns
            trainable=hopfield_trainable
        )
        
    def forward(self, quant_image, quant_lnn=None):
        """
        Args:
            quant_image: Tensor of shape (batch_size, 512, H, W).
            quant_lnn:   Tensor of shape (batch_size, 16, H_lnn, W_lnn) [optional].
            
        Returns:
            Tensor of shape (batch_size, 512, H, W).
        """
        B, _, H_image, W_image = quant_image.shape

        if self.use_lnn and quant_lnn is not None:
            # Resize LNN embeddings to match the image resolution (e.g., 16x16)
            quant_lnn_resized = F.interpolate(
                quant_lnn,
                size=(H_image, W_image),
                mode="bilinear",
                align_corners=False
            )  # shape: (B, 16, H_image, W_image)

            # Concatenate along channel dimension
            combined = torch.cat([quant_image, quant_lnn_resized], dim=1)  # (B, 512+16, H, W)
            # Project to shared_dim
            combined = self.combine_proj(combined)  # (B, shared_dim, H, W)

            # Flatten spatial dimensions: (B, shared_dim, H*W) -> (B, H*W, shared_dim)
            combined_flat = combined.view(B, combined.shape[1], -1).permute(0, 2, 1)

            # Pass through Hopfield layer
            hopfield_output = self.hopfield(combined_flat)  # (B, H*W, 512)

            # Reshape back to (B, 512, H, W)
            output = hopfield_output.permute(0, 2, 1).view(B, 512, H_image, W_image)

        else:
            # Only use image embeddings
            image_projected = self.image_proj(quant_image)  # (B, shared_dim, H, W)
            # Flatten spatial dimensions
            image_flat = image_projected.view(B, image_projected.shape[1], -1).permute(0, 2, 1)  # (B, H*W, shared_dim)
            
            # Pass through Hopfield layer
            hopfield_output = self.hopfield(image_flat)  # (B, H*W, 512)
            
            # Reshape back
            output = hopfield_output.permute(0, 2, 1).view(B, 512, H_image, W_image)

        return output
