import torch
from transformers import AutoImageProcessor, AutoModel

class ImageEmbeddingService:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.model.eval()
        self.storage = {}

    def process_image(self, image):
        inputs = self.processor(image.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        emb = outputs.pooler_output.numpy()
        return emb

    def add_image(self, image_id, image_emb, image_class):
        self.storage[image_id] = {
            "vector": image_emb.tolist(),
            "class": image_class
        }

    def delete_images(self):
        self.storage.clear()
