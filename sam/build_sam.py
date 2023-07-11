# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from pathlib import Path
import urllib.request

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize
import numpy as np

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # Load the checkpoint if we want to fine-tune
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    sam.load_state_dict(state_dict)    
    
    return sam



def prepare_sam(checkpoint=None, model_type="base"):
    """
    Prepare the SAM model for inference/training

    Args:
        checkpoint (str): Path to the checkpoint to load. If None, the default
            checkpoint will be downloaded.

    Returns:
        sam (Sam): The SAM model.
    """

    # build_sam = build_sam_vit_l(checkpoint=checkpoint) # By Default we use ViT-L

    # Load the checkpoint if we want to fine-tune
    if checkpoint is not None:
        # Download the checkpoint if it does not exist
        checkpoint = Path(checkpoint)
        if checkpoint.name == "sam_vit_b_01ec64.pth" and model_type == "base":
            if not checkpoint.exists():
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-B checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
            build_sam = build_sam_vit_b(checkpoint=checkpoint)
        elif checkpoint.name == "sam_vit_h_4b8939.pth" and model_type == "huge":
            if not checkpoint.exists():
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-H checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
            build_sam = build_sam_vit_h(checkpoint=checkpoint)
        elif checkpoint.name == "sam_vit_l_0b3195.pth" and model_type == "large": 
            if not checkpoint.exists():
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading SAM ViT-L checkpoint...")
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    checkpoint,
                )
                print(checkpoint.name, " is downloaded!")
            build_sam = build_sam_vit_l(checkpoint=checkpoint)
        else:
            if model_type == "base":
                build_sam = build_sam_vit_b(checkpoint=checkpoint)
            elif model_type == "huge":
                build_sam = build_sam_vit_h(checkpoint=checkpoint)
            elif model_type == "large":
                build_sam = build_sam_vit_l(checkpoint=checkpoint)

    return build_sam


class SAM_Model():

    def __init__(self, checkpoint, model_type, device):
        self.device = device
        self.model = prepare_sam(checkpoint, model_type)
        self.model.to(self.device)

    def predict(self, image, box_prompts):
        image = np.array(image)
        sam_transform = ResizeLongestSide(self.model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        resize_img = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        resize_img = self.model.preprocess(resize_img[None,:,:,:])

        box_prompts = sam_transform.apply_boxes(np.array(box_prompts), tuple(image.shape[1:]))
        box_prompts = torch.as_tensor(box_prompts).to(self.device)

        self.model.eval()
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(resize_img)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_prompts,
                masks=None,
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False, # Since multi mask is enabled we will get 3 masks
            ) 
    
        return low_res_masks, iou_predictions
    
    def postprocess(self, low_res_masks, original_image_size):
        upscaled_masks = self.model.postprocess_masks(low_res_masks, (1024, 1024), original_image_size)
        high_res_masks = normalize(threshold(upscaled_masks, 0.0, 0))

        return high_res_masks