import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def filter_boxes(predictions, ensemble, beta, threshold):
    ensemble = ensemble.astype(np.uint8)
    filtered_boxes = []

    for i in range (len(predictions['boxes'])):
        xmin = predictions['boxes'][i][0]
        ymin = predictions['boxes'][i][1]
        xmax = predictions['boxes'][i][2]
        ymax = predictions['boxes'][i][3]
        
        box_mask = np.zeros((ensemble.shape[0], ensemble.shape[1]))
        box_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        num_ones_box = np.count_nonzero(box_mask)
        res = ensemble * box_mask
        num_ones_intersection = np.count_nonzero(res)

        _overlap = num_ones_intersection / num_ones_box

        combined_threshold = _overlap * beta + (1 - beta) * predictions['scores'][i]

        if combined_threshold > threshold:
            filtered_boxes.append(predictions['boxes'][i])
        
    return filtered_boxes

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_iou_matrix(target_masks, predicted_masks):
    num_target_masks = len(target_masks)
    num_predicted_masks = len(predicted_masks)

    print("num_target_masks: ", num_target_masks)
    print("num_predicted_masks: ", num_predicted_masks)

    iou_matrix = np.zeros((num_target_masks, num_predicted_masks))
    for i in range(num_target_masks):
        for j in range(num_predicted_masks):
            iou_matrix[i, j] = iou(target_masks[i], predicted_masks[j])

    return iou_matrix

def calculate_iou(target_masks, predicted_masks):
    iou_matrix = calculate_iou_matrix(target_masks, predicted_masks)
    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    print("row_ind: ", row_ind)
    print("col_ind: ", col_ind)

    total_iou = 0.0
    for i, j in zip(row_ind, col_ind):
        total_iou += iou_matrix[i, j]

    average_iou = total_iou / len(row_ind)
    return average_iou, iou_matrix[row_ind, col_ind]

def calculate_precision_recall(matched_ious, num_pred_instances, num_gt_instances, threshold=0.5):
    num_true_positives = len(matched_ious[matched_ious >= threshold])
    num_false_positives = num_pred_instances - num_true_positives
    num_false_negatives = num_gt_instances - num_true_positives
    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    
    return precision, recall