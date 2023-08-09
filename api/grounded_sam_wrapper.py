import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Grounded_Sam_wrapper:
    def __init__(self, config, grounded_checkpoint, sam_checkpoint, sam_hq_checkpoint, use_sam_hq, output_dir, box_threshold, text_threshold, device ) -> None:
        self.config_file = config  # change the path of the model config file
        self.grounded_checkpoint = grounded_checkpoint  # change the path of the model
        self.sam_checkpoint = sam_checkpoint
        self.sam_hq_checkpoint = sam_hq_checkpoint
        self.use_sam_hq = use_sam_hq
        #image_path = input_image
        #text_prompt = text_prompt
        self.output_dir = output_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.predictor = None
        self.model = None
        
    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]
        
        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
        
    def load(self):
         # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # load model
        self.model = self.load_model(self.config_file, self.grounded_checkpoint, device=self.device)
        
        # initialize SAM
        if self.use_sam_hq:
            self.predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        

    def run(self, image_path, text_prompt):
                
        # load image
        image_pil, image = self.load_image(image_path)
        
        image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.model, image, text_prompt, self.box_threshold, self.text_threshold, device=self.device
        )
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)

        size = image_pil.size
        
        H, W = size[1], size[0]
        print("height {}".format(H))
        print("width {}".format(W))
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        
        return masks, pred_phrases
        
        
        