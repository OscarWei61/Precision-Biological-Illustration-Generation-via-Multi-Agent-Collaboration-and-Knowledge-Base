"""
CLIP-Score evaluator.
Uses open_clip (ViT-B/32) to compute cosine similarity between image and text.
"""

import torch
import open_clip
import numpy as np
from PIL import Image


class CLIPScorer:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tensor = self.tokenizer([text]).to(self.device)

        img_feat = self.model.encode_image(img_tensor)
        txt_feat = self.model.encode_text(text_tensor)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        return float((img_feat * txt_feat).sum().item())
