import torch
from torch.utils.data import DataLoader

import numpy as np

from rpn.build_rpn import RPN_Model
from sam.build_sam import SAM_Model
from transformers import pipeline

from data_builder.build_dataset import PlanetscopeDataset

import os
import time
import json

from eval_utils import filter_boxes, calculate_iou, calculate_precision_recall

import pandas as pd
from rpn import _transforms as T

# Define the image transforms
def get_transform(image_enhancement="FALSE"):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    if image_enhancement == "TRUE":
        transforms.append(T.ContrastBasedAdaptiveGammaCorrection())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

# Read and Print the evaluation config from the JSON file
# Create a folder with unix timestamp
root_path = os.getcwd() + '/results/' + str(int(time.time())) + '/'
os.mkdir(root_path)

with open('eval_config.json', 'r') as f:
    eval_config = json.load(f)


print("Evaluation Configuration")
print(json.dumps(eval_config, indent=1))

# Set the device and load the dataset for evaluation
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_test = PlanetscopeDataset(eval_config['DATASET_PATH'], get_transform(image_enhancement=eval_config['IMAGE_ENHANCEMENT']))

torch.manual_seed(1)
indices = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(len(indices)*0.2):])

# Load the proper model to be used for evaluation
if eval_config['MODEL'] == 'SAT-SAM':
    rpn_model = RPN_Model(eval_config['RPN_MODEL_PATH'], 2, device, eval_config['TRAIN_TYPE'])
    sam_model = SAM_Model(eval_config['SAM_MODEL_PATH'], 'large', device)
elif eval_config['MODEL'] == 'SAM':
    vanilla_sam_model = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)
elif eval_config['MODEL'] == 'MASKRCNN':
    maskrcnn_model = RPN_Model(eval_config['RPN_MODEL_PATH'], 2, device, eval_config['TRAIN_TYPE']) #Load the Pre-Trained MaskRCNN model

# Run the Evaluation
results = pd.DataFrame(columns=['parcel_id', 'parcel_path', 'gt_mask_ct', 'pred_mask_ct', 'mean_iou', 'p_50', 'r_50', 'p_70', 'r_70', 'p_90', 'r_90'])

for i, (id, sam_image, rpn_image, target, ensemble, path)  in enumerate(dataset_test): 
    try:
        if eval_config['MODEL'] == 'SAT-SAM':
            print('SAT-SAM')
            rpn_image = rpn_image.squeeze(0).to(device)  
            predictions = rpn_model.predict(rpn_image)
            predictions = rpn_model.postprocess(predictions, nms_threshold=eval_config['NMS_THRESHOLD'], score_threshold=eval_config['PRED_CONFIDENCE_THRESHOLD'])

            filtered_predictions = filter_boxes(predictions, ensemble, eval_config['ENSEMBLE_BOX_BETA'], eval_config['ENSEMBLE_BOX_OVERLAP_THRESHOLD'])

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

        print("Image Id: ", id, " Average IoU Score: ", iou_score)
        print("IoU Matrix: ", iou_matrix)
        print("Precision: ", p_50, p_70, p_90)
        print("Recall: ", r_50, r_70, r_90)

        results.loc[i] = [id, path, len(target['masks']), len(pred_masks), iou_score, p_50, r_50, p_70, r_70, p_90, r_90]
    except:
        print("Error in image: ", id)
        continue

mean_IoU = results['mean_iou'].mean()
eval_config['MEAN_IOU'] = mean_IoU

# Save the results to a CSV file along with the evaluation config
results.to_csv(root_path + 'results', index=False)

# save model and configuration
with open(root_path + 'eval_config.json', 'w') as f:
    json.dump(eval_config, f, indent=1)
    f.close()
print("Evaluation Configuration saved to " + root_path + 'train_config.json') 