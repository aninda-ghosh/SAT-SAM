import torch
from torch.utils.data import DataLoader

import numpy as np

from rpn.build_rpn import RPN_Model
from sam.build_sam import SAM_Model
from transformers import pipeline

from data_builder.build_dataset import PlanetscopeDataset

import json

from eval_utils import filter_boxes, calculate_iou, calculate_precision_recall

import pandas as pd


with open('eval_config.json', 'r') as f:
    eval_config = json.load(f)

print("Evaluation Configuration")
print(json.dumps(eval_config, indent=1))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = PlanetscopeDataset(eval_config['DATASET_PATH'], train=False)

if eval_config['MODEL'] == 'SAT-SAM':
    rpn_model = RPN_Model(eval_config['RPN_MODEL_PATH'], 2, device)
    sam_model = SAM_Model(eval_config['SAM_MODEL_PATH'], 'large', device)
elif eval_config['MODEL'] == 'SAM':
    vanilla_sam_model = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)
elif eval_config['MODEL'] == 'MASKRCNN':
    maskrcnn_model = RPN_Model(eval_config['RPN_MODEL_PATH'], 2, device) #Load the Pre-Trained MaskRCNN model
    # maskrcnn_model = RPN_Model(None, 2, device)   #Load the Vanilla MaskRCNN model


results = pd.DataFrame(columns=['parcel_id', 'gt_mask_ct', 'pred_mask_ct', 'mean_iou', 'p_50', 'r_50', 'p_70', 'r_70', 'p_90', 'r_90'])

for i, (sam_image, rpn_image, target, ensemble)  in enumerate(dataset): 
    try:
        if eval_config['MODEL'] == 'SAT-SAM':
            print('SAT-SAM')
            rpn_image = rpn_image.squeeze(0).to(device)  
            predictions = rpn_model.predict(rpn_image)
            predictions = rpn_model.postprocess(predictions, nms_threshold=eval_config['NMS_THRESHOLD'], score_threshold=eval_config['PRED_CONFIDENCE_THRESHOLD'])

            filtered_predictions = filter_boxes(predictions, ensemble, eval_config['ENSEMBLE_BOX_OVERLAP_THRESHOLD'])

            low_res_masks, iou_predictions = sam_model.predict(sam_image, filtered_predictions)
            high_res_masks = sam_model.postprocess(low_res_masks, tuple(sam_image.size))
            pred_masks = high_res_masks.squeeze().cpu().numpy()
        
        elif eval_config['MODEL'] == 'SAM':
            print('SAM')
            outputs = vanilla_sam_model(sam_image, points_per_batch=32)
            pred_masks = outputs["masks"]
        
        elif eval_config['MODEL'] == 'MASKRCNN':
            print('MASKRCNN')
            rpn_image = rpn_image.squeeze(0).to(device)
            predictions = maskrcnn_model.predict(rpn_image)
            predictions = maskrcnn_model.postprocess(predictions, nms_threshold=eval_config['NMS_THRESHOLD'], score_threshold=eval_config['PRED_CONFIDENCE_THRESHOLD'])
            pred_masks = predictions['masks']

        iou_score, iou_matrix = calculate_iou(target_masks=np.array(target['masks']), predicted_masks=np.array(pred_masks))
        
        p_50, r_50 = calculate_precision_recall(iou_matrix, len(pred_masks), len(target['masks']), threshold=0.5)
        p_70, r_70 = calculate_precision_recall(iou_matrix, len(pred_masks), len(target['masks']), threshold=0.7)
        p_90, r_90 = calculate_precision_recall(iou_matrix, len(pred_masks), len(target['masks']), threshold=0.9)
        
        # Round off the values to 2 decimal places
        iou_score = round(iou_score, 2)
        p_50, r_50 = round(p_50, 2), round(r_50, 2)
        p_70, r_70 = round(p_70, 2), round(r_70, 2)
        p_90, r_90 = round(p_90, 2), round(r_90, 2)

        print("Image Id: ", i, " Average IoU Score: ", iou_score)
        print("IoU Matrix: ", iou_matrix)
        print("Precision: ", p_50, p_70, p_90)
        print("Recall: ", r_50, r_70, r_90)

        results.loc[i] = [i, len(target['masks']), len(pred_masks), iou_score, p_50, r_50, p_70, r_70, p_90, r_90]
    except:
        print("Error in image: ", i)
        continue

# Input: dataframe, model name, dataset name
# Output: csv file

def save_csv(df, model_name, dataset_name):
    df.to_csv('{}_{}.csv'.format(model_name, dataset_name), index=False)

save_csv(results, eval_config['MODEL'], eval_config['REGION'])