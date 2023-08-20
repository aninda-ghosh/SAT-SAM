import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def filter_boxes_predictions(predictions, ensemble, beta, threshold):
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

        if combined_threshold >= threshold:
            filtered_boxes.append(predictions['boxes'][i])
        
    return filtered_boxes

def filter_boxes(bboxes, ensemble, threshold):
    ensemble = ensemble.astype(np.uint8)
    filtered_boxes = []

    for i in range (len(bboxes)):
        xmin = bboxes[i][0]
        ymin = bboxes[i][1]
        xmax = bboxes[i][2]
        ymax = bboxes[i][3]
        
        box_mask = np.zeros((ensemble.shape[0], ensemble.shape[1]))
        box_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        num_ones_box = np.count_nonzero(box_mask)
        res = ensemble * box_mask
        num_ones_intersection = np.count_nonzero(res)

        _overlap = num_ones_intersection / num_ones_box

        if _overlap >= threshold:
            filtered_boxes.append(bboxes[i])
        
    return filtered_boxes

def filter_masks(predictions, ensemble, threshold):
    ensemble = ensemble.astype(np.uint8)
    filtered_masks = []
    
    for mask in predictions:
        num_ones_box = np.count_nonzero(mask)
        res = ensemble * mask
        num_ones_intersection = np.count_nonzero(res)

        _overlap = num_ones_intersection / num_ones_box

        if _overlap > threshold:
            filtered_masks.append(mask)
        
    return filtered_masks

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


# To Generate the Random Points and Bounding Boxes using https://www.jasondavies.com/poisson-disc/ algorithm
def generate_random_bounding_boxes(image_width, image_height, number_boxes, min_distance, box_percentage):
    def generate_random_points(image_width, image_height, min_distance, num_points, k=30):
        image_size = (image_width, image_height)
        cell_size = min_distance / np.sqrt(2)
        grid_width = int(np.ceil(image_width / cell_size))
        grid_height = int(np.ceil(image_height / cell_size))
        grid = np.empty((grid_width, grid_height), dtype=np.int32)
        grid.fill(-1)

        points = []
        active_points = []

        def generate_random_point():
            return np.random.uniform(0, image_width), np.random.uniform(0, image_height)

        def get_neighboring_cells(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = []
            for i in range(max(0, x_index - 2), min(grid_width, x_index + 3)):
                for j in range(max(0, y_index - 2), min(grid_height, y_index + 3)):
                    cells.append((i, j))

            return cells

        def is_point_valid(point):
            x, y = point
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                return False

            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = get_neighboring_cells(point)
            for cell in cells:
                if grid[cell] != -1:
                    cell_points = points[grid[cell]]
                    if np.any(np.linalg.norm(np.array(cell_points) - np.array(point), axis=None) < min_distance):
                        return False

            return True

        def add_point(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            points.append(point)
            index = len(points) - 1
            grid[x_index, y_index] = index
            active_points.append(point)

        start_point = generate_random_point()
        add_point(start_point)

        while active_points and len(points) < num_points:
            random_index = np.random.randint(len(active_points))
            random_point = active_points[random_index]
            added_new_point = False

            for _ in range(k):
                angle = 2 * np.pi * np.random.random()
                radius = min_distance + min_distance * np.random.random()
                new_point = (random_point[0] + radius * np.cos(angle), random_point[1] + radius * np.sin(angle))
                if is_point_valid(new_point):
                    add_point(new_point)
                    added_new_point = True

            if not added_new_point:
                active_points.pop(random_index)

        return points
    

    points = generate_random_points(image_width, image_height, min_distance, number_boxes)
    
    
    box_width = int(image_width * box_percentage)
    box_height = int(image_height * box_percentage)

    bounding_boxes = []
    for point in points:
        x = int(point[0] - box_width / 2)
        y = int(point[1] - box_height / 2)

        # Adjust the coordinates to keep the bounding box within the image
        x = max(0, min(x, image_width - box_width))
        y = max(0, min(y, image_height - box_height))

        bounding_boxes.append([x, y, x+box_width, y+box_height])

    return bounding_boxes


def generate_uniform_spaced_bounding_boxes(image_width, image_height, box_size):
    stride = int(0.6 * box_size)
    num_boxes_horizontal = (image_width - box_size) // stride + 1
    num_boxes_vertical = (image_height - box_size) // stride + 1

    bounding_boxes = []
    for i in range(num_boxes_horizontal):
        for j in range(num_boxes_vertical):
            start_x = i * stride
            start_y = j * stride
            end_x = start_x + box_size
            end_y = start_y + box_size

            # Adjust for boxes extending beyond image boundaries
            end_x = min(end_x, image_width)
            end_y = min(end_y, image_height)

            bounding_boxes.append((start_x, start_y, end_x, end_y))

    return bounding_boxes