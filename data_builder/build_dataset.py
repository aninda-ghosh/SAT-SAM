import torch
import torch.utils.data

import numpy as np
from PIL import Image

import data_builder.transforms as T

class PlanetscopeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(self.root+'data.txt') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            lines = [x.split(',') for x in lines]

        # Create separate lists for image, mask, and geojson file paths
        img_path = []
        mask_path = []
        geojson_path = []
        ensemble_path = []

        for i in range(len(lines)):
            img_path.append(self.root+lines[i][0])
            mask_path.append(self.root+lines[i][1])
            geojson_path.append(self.root+lines[i][2])
            ensemble_path.append(self.root+lines[i][3])

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = img_path
        self.masks = mask_path
        self.geojson = geojson_path
        self.ensemble = ensemble_path

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        ensemble_path = self.ensemble[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes, dtype=np.float32)

        labels = np.ones((num_objs,), dtype=np.int64)        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = np.zeros((num_objs,), dtype=np.int64)

        temptargets = {}
        temptargets["boxes"] = boxes
        temptargets["labels"] = labels
        temptargets["masks"] = masks
        temptargets["image_id"] = idx
        temptargets["area"] = area
        temptargets["iscrowd"] = iscrowd


        #Remove the entries with area 0
        target = {}
        target["boxes"] = []
        target["labels"] = []
        target["masks"] = []
        target["image_id"] = idx
        target["area"] = []
        target["iscrowd"] = []

        for i in range(len(temptargets["area"])):
            if temptargets["area"][i] != 0:
                target["boxes"].append(temptargets["boxes"][i])
                target["labels"].append(temptargets["labels"][i])
                target["masks"].append(temptargets["masks"][i])
                target["area"].append(temptargets["area"][i])
                target["iscrowd"].append(temptargets["iscrowd"][i])

        target["boxes"] = torch.as_tensor(np.array(target["boxes"], dtype=np.float32))
        target["labels"] = torch.as_tensor(np.array(target["labels"], dtype=np.int64))
        target["masks"] = torch.as_tensor(np.array(target["masks"], dtype=np.uint8))
        target["image_id"] = torch.as_tensor(target["image_id"])
        target["area"] = torch.as_tensor(np.array(target["area"], dtype=np.float32))
        target["iscrowd"] = torch.as_tensor(np.array(target["iscrowd"], dtype=np.int64))


        # Read the ensemble layer and convert to binary mask
        ensemble = np.load(ensemble_path)


        if self.transforms is not None:
            rpn_img, target = self.transforms(img, target)

        return idx, img, rpn_img, target, ensemble, img_path

    def __len__(self):
        return len(self.imgs)