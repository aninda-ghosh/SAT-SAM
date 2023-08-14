import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

from _engine import train_one_epoch, evaluate
import _utils
import _transforms as T

from _dataset import ParcelDataset

import os
import time
import json

      
def get_instance_segmentation_model(num_classes, train_type):
    # load an instance segmentation model pre-trained on COCO
    if train_type == 'FINETUNE':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    elif train_type == 'SCRATCH':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transform(train, image_enhancement="FALSE"):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    if image_enhancement == "TRUE":
        transforms.append(T.ContrastBasedAdaptiveGammaCorrection())
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    #Create a folder with unix timestamp
    root_path = os.getcwd() + '/checkpoints/' + str(int(time.time())) + '/'
    os.mkdir(root_path)

    with open('train_config.json', 'r') as f:
        train_config = json.load(f)
    
    print("Training Configuration")
    print(json.dumps(train_config, indent=1))    

           

    # use our dataset and defined transformations
    dataset = ParcelDataset(train_config['DATASET_PATH'], get_transform(train=True, image_enhancement=train_config['IMAGE_ENHANCEMENT']))
    dataset_test = ParcelDataset(train_config['DATASET_PATH'], get_transform(train=False))

    # dataset2 = ParcelDataset(train_config['DATASET2_PATH'], get_transform(train=True, image_enhancement=train_config['IMAGE_ENHANCEMENT']))
    # dataset2_test = ParcelDataset(train_config['DATASET2_PATH'], get_transform(train=False))

    # dataset = ConcatDataset([dataset1, dataset2])
    # dataset_test = ConcatDataset([dataset1_test, dataset2_test])

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-int(len(indices)*0.2)])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(len(indices)*0.2):])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=_utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=_utils.collate_fn)
    

    torch.cuda.empty_cache()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_instance_segmentation_model(train_config['NO_OF_CLASSES'], train_config['TRAIN_TYPE'])

    if train_config['LOAD_MODEL'] == "TRUE":
        model.load_state_dict(torch.load(train_config['MODEL_PATH']))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=train_config['LEARNING_RATE'], momentum=train_config['MOMENTUM'], weight_decay=train_config['WEIGHT_DECAY'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config['LEARNING_RATE_STEP'], gamma=train_config['LEARNING_RATE_GAMMA'])
    
    best_epoch_loss = 100000

    for epoch in range(train_config['EPOCHS']):
        # train for one epoch, printing every 10 iterations
        train_metric = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

         #Print the train metric loss
        print('Train Loss: ' + str(train_metric.meters['loss'].avg))
        
        if train_metric.meters['loss'].avg < best_epoch_loss:
            best_epoch_loss = train_metric.meters['loss'].avg
            torch.save(model.state_dict(), root_path + 'rpn_model_' + str(round(best_epoch_loss,2)) + '.pth')

        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        torch.cuda.empty_cache()

    # save model and configuration
    with open(root_path + 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=1)
        f.close()
    print("Training Configuration saved to " + root_path + 'train_config.json') 
    torch.save(model.state_dict(), root_path + 'rpn_model_' + str(round(best_epoch_loss,2)) + '.pth')

if __name__ == '__main__':
    main()