import torch
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

def postprocess_batch(outputs, 
                      sem_thresh=0.5, 
                      seed_thresh=0.5, 
                      min_distance=5):
    """
    Convert model output (B,4,H,W) into instance label masks (B,H,W).
    
    Args:
        outputs (torch.Tensor): shape (B,4,H,W)
            - Channel 0: Semantic (logit or probability) for foreground
            - Channel 1: Seed map for instance centers
            - Channel 2,3: (Optional) embeddings or other features not directly used here
        sem_thresh (float): Threshold for semantic channel
        seed_thresh (float): Threshold for seed map peaks
        min_distance (int): Minimum distance for peak_local_max (helps separate nearby peaks)
    
    Returns:
        List of numpy arrays [ (H, W), ... ] with integer instance IDs for each image.
    """
    
    # If channel 0 is a logit, apply sigmoid to convert to probability
    # If channel 1 is also a logit for seeds, apply sigmoid similarly
    # For simplicity, let's assume channel 0,1 are raw logits
    # Adjust if your model already outputs probabilities
    semantic_logit = outputs[:, 0, ...]
    seed_logit     = outputs[:, 1, ...]
    
    semantic_prob = torch.sigmoid(semantic_logit)
    seed_prob     = torch.sigmoid(seed_logit)
    
    # Convert to CPU numpy for processing with skimage/scipy
    semantic_prob = semantic_prob.detach().cpu().numpy()
    seed_prob     = seed_prob.detach().cpu().numpy()
    
    batch_size = outputs.shape[0]
    
    instance_label_maps = []
    
    for b in range(batch_size):
        # 1) Threshold semantic channel -> foreground mask
        foreground_mask = semantic_prob[b] > sem_thresh
        
        # 2) Convert seed channel to a mask for local maxima detection
        seed_image = seed_prob[b]
        
        # 3) Distance transform on the foreground to help watershed
        dist_map = distance_transform_edt(foreground_mask)
        
        # 4) Find local maxima in seed map (only inside foreground)
        #    Using skimage's peak_local_max
        local_max_coords = peak_local_max(
            seed_image, 
            min_distance=min_distance,
            threshold_abs=seed_thresh,
            labels=foreground_mask
        )
        
        # Create a marker image for watershed: each local maximum is a unique label
        markers = np.zeros_like(seed_image, dtype=np.int32)
        for i, (r, c) in enumerate(local_max_coords, start=1):
            markers[r, c] = i
        
        # 5) Run watershed
        #    Watershed will assign each foreground pixel to the nearest marker region
        #    (or 0 if not reachable). We use -dist_map so that watersheds expand from peaks.
        instance_labels = watershed(
            -dist_map, 
            markers, 
            mask=foreground_mask
        )
        
        # NOTE: If two peaks are extremely close or if the seed is weak, 
        # you might see merges or splits. Adjust min_distance, seed_thresh, etc.
        
        # (Optionally) If you have embeddings, you can refine instance boundaries 
        # or do further clustering to separate touching instances.
        # For a minimal example, we just rely on watershed + seed map.
        
        instance_label_maps.append(instance_labels)
    
    return instance_label_maps