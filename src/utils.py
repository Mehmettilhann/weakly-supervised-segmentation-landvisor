import torch
import numpy as np

def calculate_iou(preds, labels, num_classes, ignore_index=255):
    """ 
    Calculates the Intersection over Union (IoU) metric. 
    
    Args:
        preds: Model predictions (Logits)
        labels: Ground truth labels
        num_classes: Total number of classes
        ignore_index: Unlabeled pixels to be excluded from the calculation

    Returns:
        mean_iou: Average IoU score of all classes  

    """

    preds = torch.argmax(preds, dim=1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_inds = preds == cls
        target_inds = labels == cls

        if not np.any(target_inds):
            continue

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    
    if len(valid_ious) == 0:
        return 0.0
        
    mean_iou = np.mean(valid_ious)
    return mean_iou