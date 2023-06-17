from src.service.image_embedding_service import ImageEmbeddingService
from src.service.similarity_search_service import SimilaritySearchService
from src.util.pil_images import get_pil_images

class ImageService:
    def __init__(self):
        self.embedding_service = ImageEmbeddingService()
        self.similarity_search_service = SimilaritySearchService(self.embedding_service)

    async def upload_image(self, image , image_class):
        image_id = image.filename
        image_emb = self.embedding_service.process_image(await get_pil_images(image))
        self.similarity_search_service.add_image(image_id, image_emb, image_class)

    async def search_similar_image(self, image):
        result = self.similarity_search_service.search_similar_image(await get_pil_images(image))
        return result

    def delete_images(self):
        self.similarity_search_service.delete_images()
