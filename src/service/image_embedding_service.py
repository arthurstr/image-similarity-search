import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

class ImageEmbeddingService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window12-384").to(self.device)
        self.model.eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        emb = outputs.pooler_output.cpu().numpy()
        return emb
