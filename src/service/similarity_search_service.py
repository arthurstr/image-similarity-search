import faiss
import numpy as np

class SimilaritySearchService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.index = faiss.IndexFlatL2(1024)

    def add_index(self, emb):
        self.index.add(emb)

    def search_similar_image(self, image):
        image_emb = self.embedding_service.process_image(image)
        D, I = self.index.search(image_emb, k=1)

        nearest_neighbor = {}
        nearest_neighbor["id"] = list(self.embedding_service.storage.keys())[I[0][0]]
        nearest_neighbor["class"] = self.embedding_service.storage[nearest_neighbor["id"]]["class"]
        nearest_neighbor["distance"] = float(D[0][0])

        return nearest_neighbor