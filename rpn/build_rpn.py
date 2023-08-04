import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights

class RPN_Model():

    def __init__(self, checkpoint, num_classes, device):
        self.device = device

        # load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        # get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))

        self.model.to(self.device)
    
    def predict(self, image):
        image = torch.as_tensor(image, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model([image])

        # Convert the prediction to a cpu element
        predictions = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in predictions]

        predictions = {
            'boxes': predictions[0]['boxes'],
            'scores': predictions[0]['scores'],
            'masks': predictions[0]['masks']
        }

        return predictions
        
    def postprocess(self, predictions, nms_threshold=0.8, score_threshold=0.5):
        #Perform Non-Maximum Suppression
        keep = torchvision.ops.nms(predictions['boxes'], predictions['scores'], nms_threshold)
        predictions['boxes'] = predictions['boxes'][keep]    
        predictions['scores'] = predictions['scores'][keep]
        predictions['masks'] = predictions['masks'][keep]

        #Drop the prediction with a score lower than 0.5
        keep = predictions['scores'] > score_threshold
        predictions['boxes'] = predictions['boxes'][keep]
        predictions['scores'] = predictions['scores'][keep]
        predictions['masks'] = predictions['masks'][keep]

        return predictions