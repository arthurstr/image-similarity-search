from fastapi import UploadFile
from src.service.embedding_service import ImageEmbeddingService
from src.service.body import SimilaritySearchService
from src.util.image_utils import decode

class ImageSimilaritySearchService :
    def __init__(self):
        self.embedding_service = ImageEmbeddingService()
        self.similarity_search_service = SimilaritySearchService()

    def add(self, image: UploadFile , image_class : str) -> None:
        image_id = image.filename
        image_emb = self.embedding_service.embed(decode(image))
        self.similarity_search_service.add(image_id, image_emb, image_class)

    def search(self, image: UploadFile) -> dict:
        image_emb = self.embedding_service.embed(decode(image))
        result = self.similarity_search_service.search(image_emb)
        return result

    def reset(self) -> None:
        self.similarity_search_service.reset()
