from fastapi import UploadFile
from src.service.image_embedding_service import ImageEmbeddingService
from src.service.similarity_search_service import SimilaritySearchService
from src.util.image_utils import decode_image
from typing import Dict

class ImageSimilaritySearchService :
    def __init__(self):
        self.embedding_service = ImageEmbeddingService()
        self.similarity_search_service = SimilaritySearchService()

    def add_image(self, image: UploadFile , image_class : str) -> None:
        image_id = image.filename
        image_emb = self.embedding_service.process_image(decode_image(image))
        self.similarity_search_service.add_image(image_id, image_emb, image_class)

    def search_similar_image(self, image: UploadFile) -> Dict[str, str]:
        image_emb = self.embedding_service.process_image(decode_image(image))
        result = self.similarity_search_service.search_similar_image(image_emb)
        return result

    def reset_images(self) -> None:
        self.similarity_search_service.delete_images()
