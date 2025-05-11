import torch
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes.
    
    Args:
        box1 (torch.Tensor or list): First box in format [x1, y1, x2, y2]
        box2 (torch.Tensor or list): Second box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Convert to numpy if tensor
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_precision_recall(detections, ground_truth, iou_threshold=0.5):
    """Calculate precision and recall for object detection.
    
    Args:
        detections (list): List of detections, each detection is [class_id, confidence, x1, y1, x2, y2]
        ground_truth (list): List of ground truth boxes, each is [class_id, x1, y1, x2, y2]
        iou_threshold (float): IoU threshold for considering a detection as correct
        
    Returns:
        tuple: (precision, recall)
    """
    # Sort detections by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    
    # Create a copy of ground truth to mark matched boxes
    gt_matched = [False] * len(ground_truth)
    
    # Process each detection
    for detection in detections:
        class_id, _, *bbox = detection
        
        # Find best matching ground truth box
        best_iou = 0
        best_idx = -1
        
        for i, gt in enumerate(ground_truth):
            gt_class_id, *gt_bbox = gt
            
            # Skip if class doesn't match or already matched
            if gt_class_id != class_id or gt_matched[i]:
                continue
            
            # Calculate IoU
            iou = calculate_iou(bbox, gt_bbox)
            
            # Update best match
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_idx = i
        
        # Check if a match was found
        if best_idx >= 0:
            true_positives += 1
            gt_matched[best_idx] = True
        else:
            false_positives += 1
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    
    return precision, recall

def calculate_map(detections_list, ground_truth_list, num_classes, iou_threshold=0.5):
    """Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        detections_list (list): List of detections for each image
        ground_truth_list (list): List of ground truth boxes for each image
        num_classes (int): Number of classes
        iou_threshold (float): IoU threshold for considering a detection as correct
        
    Returns:
        tuple: (mAP, AP_per_class)
    """
    # Initialize AP for each class
    AP_per_class = {}
    
    # Process each class
    for class_id in range(1, num_classes):  # Skip background class (0)
        # Collect all detections and ground truths for this class
        all_detections = []
        all_ground_truths = []
        
        for img_idx, (detections, ground_truths) in enumerate(zip(detections_list, ground_truth_list)):
            # Filter detections for this class
            class_detections = [d for d in detections if d[0] == class_id]
            all_detections.extend([(img_idx, d[1], d[2:]) for d in class_detections])  # (img_idx, confidence, bbox)
            
            # Filter ground truths for this class
            class_ground_truths = [gt for gt in ground_truths if gt[0] == class_id]
            all_ground_truths.extend([(img_idx, gt[1:]) for gt in class_ground_truths])  # (img_idx, bbox)
        
        # Skip if no ground truths for this class
        if len(all_ground_truths) == 0:
            AP_per_class[class_id] = 0
            continue
        
        # Sort detections by confidence (descending)
        all_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize counters
        tp = np.zeros(len(all_detections))
        fp = np.zeros(len(all_detections))
        
        # Create a dictionary to store ground truths per image
        gt_per_img = defaultdict(list)
        gt_matched = defaultdict(list)
        
        for img_idx, bbox in all_ground_truths:
            gt_per_img[img_idx].append(bbox)
            gt_matched[img_idx].append(False)
        
        # Process each detection
        for i, (img_idx, _, bbox) in enumerate(all_detections):
            # Skip if no ground truths for this image
            if img_idx not in gt_per_img:
                fp[i] = 1
                continue
            
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(gt_per_img[img_idx]):
                # Skip if already matched
                if gt_matched[img_idx][gt_idx]:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(bbox, gt_bbox)
                
                # Update best match
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if a match was found
            if best_gt_idx >= 0:
                gt_matched[img_idx][best_gt_idx] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / len(all_ground_truths)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        AP_per_class[class_id] = ap
    
    # Calculate mAP
    mAP = sum(AP_per_class.values()) / len(AP_per_class) if AP_per_class else 0
    
    return mAP, AP_per_class

def calculate_map_range(detections_list, ground_truth_list, num_classes, iou_range=(0.5, 0.95, 0.05)):
    """Calculate mAP over a range of IoU thresholds (e.g., mAP@[0.5:0.95]).
    
    Args:
        detections_list (list): List of detections for each image
        ground_truth_list (list): List of ground truth boxes for each image
        num_classes (int): Number of classes
        iou_range (tuple): (min_iou, max_iou, step)
        
    Returns:
        float: mAP averaged over IoU thresholds
    """
    min_iou, max_iou, step = iou_range
    iou_thresholds = np.arange(min_iou, max_iou + step, step)
    
    # Calculate mAP for each IoU threshold
    mAP_list = []
    for iou_threshold in iou_thresholds:
        mAP, _ = calculate_map(detections_list, ground_truth_list, num_classes, iou_threshold)
        mAP_list.append(mAP)
    
    # Average mAP over IoU thresholds
    mAP_avg = sum(mAP_list) / len(mAP_list) if mAP_list else 0
    
    return mAP_avg