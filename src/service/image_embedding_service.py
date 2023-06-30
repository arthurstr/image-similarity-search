import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

class ImageEmbeddingService:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.model.eval()

    def process_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        emb = outputs.pooler_output.numpy()
        return emb
