import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from scipy.optimize import linear_sum_assignment

################################################################################
# 1) Helper functions
################################################################################

def label_to_binary_masks(label_map):
    """
    Convert a 2D labeled map (H, W), where each pixel is an integer instance ID,
    into a list of binary masks (one per instance).
    label 0 => background
    """
    instance_ids = list(np.unique(label_map))
    if 0 in instance_ids:
        instance_ids.remove(0)  # remove background ID
    
    binary_masks = []
    for inst_id in instance_ids:
        mask = (label_map == inst_id)
        binary_masks.append(mask)
    return binary_masks

def iou_mask(mask1, mask2):
    """
    Compute IoU between two binary masks.
    """
    overlap = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return overlap / union

def dice_coefficient(mask1, mask2):
    """
    Compute Dice = 2 * |A ∩ B| / (|A| + |B|).
    """
    overlap = (mask1 & mask2).sum()
    size_1 = mask1.sum()
    size_2 = mask2.sum()
    return (2.0 * overlap) / (size_1 + size_2 + 1e-7)

def hausdorff_distance_95(mask1, mask2):
    """
    Compute robust Hausdorff distance (HD95).
    We:
      1) Extract the (x,y) coordinates of foreground pixels in mask1, mask2
      2) Compute pairwise distances, then take 95th percentile of the max

    For real-world performance, you may want a more optimized approach
    or rely on a library function (e.g., MONAI, SimpleITK).
    """
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    
    if len(coords1) == 0 and len(coords2) == 0:
        # Both empty masks => distance = 0
        return 0.0
    if len(coords1) == 0 or len(coords2) == 0:
        # One is empty => infinite distance in principle, but we can define a large number
        return 999.0

    # Pairwise distances
    dists_1_to_2 = np.sqrt(((coords1[:, None] - coords2[None, :]) ** 2).sum(axis=2))
    dists_2_to_1 = dists_1_to_2.T  # same matrix
    
    # 95th percentile of each distribution, then take max
    hd_1 = np.percentile(dists_1_to_2.min(axis=1), 95)
    hd_2 = np.percentile(dists_2_to_1.min(axis=1), 95)
    return max(hd_1, hd_2)

def aggregated_jaccard_index(gt_label_map, pred_label_map):
    """
    Compute the Aggregated Jaccard Index (AJI).
    AJI is commonly used in biomedical instance segmentation.

    AJI = ( sum_{k}( |G_k ∩ P(match(k))| ) ) / ( |Union of all G_k ∪ P_j| )

    Where 'match(k)' is the predicted instance that overlaps G_k the most.
    Non-matched predicted instances are also included in the denominator.
    """
    # Flatten: we only consider positive labels
    gt_instances = label_to_binary_masks(gt_label_map)
    pred_instances = label_to_binary_masks(pred_label_map)

    # ID sets
    gt_ids   = [i+1 for i in range(len(gt_instances))]
    pred_ids = [j+1 for j in range(len(pred_instances))]

    # We compute overlap areas
    overlap_areas = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, g_mask in enumerate(gt_instances):
        for j, p_mask in enumerate(pred_instances):
            overlap_areas[i, j] = (g_mask & p_mask).sum()

    try:
        
        # For each ground-truth instance, find the predicted instance with max overlap
        matched_pred_for_gt = overlap_areas.argmax(axis=1)  # best match index

        # Sum of intersections for matched
        sum_intersections = 0.0
        for i, g_mask in enumerate(gt_instances):
            j = matched_pred_for_gt[i]
            sum_intersections += overlap_areas[i, j]

        # Union of all masks
        all_gt_mask   = (gt_label_map > 0)
        all_pred_mask = (pred_label_map > 0)
        union_all = (all_gt_mask | all_pred_mask).sum()
        aji_value = sum_intersections / union_all if union_all > 0 else 0

    except Exception as e:
        return 0.0
    return aji_value

################################################################################
# 2) Main function to evaluate
################################################################################

def evaluate_instance_segmentation(
    pred_label_maps, 
    gt_label_maps, 
    iou_threshold=0.5,
):
    """
    Evaluate instance segmentation performance given a list of predicted label maps
    and ground-truth label maps.

    pred_label_maps: list of np.array, each of shape (H, W), integer IDs
    gt_label_maps:   list of np.array, each of shape (H, W), integer IDs
    iou_threshold:   float, e.g. 0.5 for matching.

    Returns a dict with:
    - dice
    - hd95
    - iou
    - precision
    - recall
    - f1
    - aji
    """
    # These lists collect per-image or per-instance values
    dice_scores      = []
    hd95_scores      = []
    iou_scores       = []
    aji_scores       = []

    # For precision/recall/F1
    # 'tp' = # matched (IoU>threshold), 'fp' = # predicted not matched, 'fn' = # GT not matched
    total_tp = 0
    total_fp = 0
    total_fn = 0

    n_images = len(pred_label_maps)
    for idx in range(n_images):
        pred_label_map = pred_label_maps[idx]
        gt_label_map   = gt_label_maps[idx]

        # --- 2.1) Convert to list of binary masks
        pred_masks = label_to_binary_masks(pred_label_map)  # List of shape [num_pred]
        gt_masks   = label_to_binary_masks(gt_label_map)    # List of shape [num_gt]
        
        num_pred = len(pred_masks)
        num_gt   = len(gt_masks)

        # --- 2.2) Build IoU matrix
        iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_pred):
                iou_matrix[i, j] = iou_mask(gt_masks[i], pred_masks[j])

        # --- 2.3) Perform bipartite matching to maximize total IoU
        # We'll use linear_sum_assignment on -iou_matrix => maximizing IoU
        # (Hungarian algorithm). Then filter matched pairs by iou_threshold.
        if num_gt > 0 and num_pred > 0:
            cost_matrix = -iou_matrix  # maximize IoU => minimize negative IoU
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
            
            matched_pairs = []
            for g_i, p_j in zip(gt_indices, pred_indices):
                curr_iou = iou_matrix[g_i, p_j]
                if curr_iou >= iou_threshold:
                    matched_pairs.append((g_i, p_j, curr_iou))
        else:
            matched_pairs = []

        # --- 2.4) Count matched/unmatched for precision/recall
        tp = len(matched_pairs)
        fp = num_pred - tp
        fn = num_gt - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # --- 2.5) Compute per-instance metrics for matched pairs
        #  (Dice, HD95, IoU)
        dice_list  = []
        hd95_list  = []
        iou_list   = []
        for (g_i, p_j, curr_iou) in matched_pairs:
            d  = dice_coefficient(gt_masks[g_i], pred_masks[p_j])
            hd = hausdorff_distance_95(gt_masks[g_i], pred_masks[p_j])
            dice_list.append(d)
            hd95_list.append(hd)
            iou_list.append(curr_iou)
        
        if dice_list:
            dice_scores.append(np.mean(dice_list))
            hd95_scores.append(np.mean(hd95_list))
            iou_scores.append(np.mean(iou_list))
        else:
            # No matched pairs => can log 0 or skip
            dice_scores.append(0.0)
            hd95_scores.append(999.0)  # or some sentinel
            iou_scores.append(0.0)

        # --- 2.6) AJI (Aggregated Jaccard Index) for the entire image
        aji_val = aggregated_jaccard_index(gt_label_map, pred_label_map)
        aji_scores.append(aji_val)

    # --- 2.7) Aggregate metrics across all images
    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0
    mean_hd95 = np.mean(hd95_scores) if len(hd95_scores) > 0 else 999
    mean_iou  = np.mean(iou_scores)  if len(iou_scores)  > 0 else 0
    mean_aji  = np.mean(aji_scores)  if len(aji_scores)  > 0 else 0

    # Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall    = total_tp / (total_tp + total_fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    # Return a dictionary of metrics
    return {
        "Dice": mean_dice,
        "HD95": mean_hd95,
        "IoU": mean_iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AJI": mean_aji,
    }
