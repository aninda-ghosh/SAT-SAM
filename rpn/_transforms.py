import random
import torch

from torchvision.transforms import functional as F

import numpy as np
import cv2


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ContrastBasedAdaptiveGammaCorrection(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        alpha=1.0
        beta=1.0

        # Convert image to float32 for calculations
        img_float = image.astype(np.float32) / 255.0
        
        # Calculate local contrast using Laplacian operator
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        local_contrast = np.abs(laplacian)
        
        # Calculate adaptive gamma based on local contrast
        adaptive_gamma = np.power(local_contrast, alpha)
        
        # Apply gamma correction
        corrected_image = np.power(img_float, adaptive_gamma) * beta
        
        # Convert back to uint8 format
        corrected_image = np.clip(corrected_image * 255.0, 0, 255).astype(np.uint8)
        
        return corrected_image, target
    
class GammaCorrection(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        image = image.astype(float)
        image = (255 * (image - np.min(image[:])) / (np.max(image[:]) - np.min(image[:]) + 0.1)).astype(float)
        image = (image + 0.5) / 256
        gamma = -1/np.nanmean(np.log(image))
        image = image**(gamma)
        return image, target