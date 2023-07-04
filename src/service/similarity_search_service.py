import faiss
import numpy as np

class SimilaritySearchService:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1024)
        self.ids = []
        self.classes = []

    def search(self, image: np.ndarray) -> dict:
        D, I = self.index.search(image, k=1)

        nearest_neighbor = {
            "id": self.ids[I[0][0]],
            "class": self.classes[I[0][0]],
            "distance": float(D[0][0])
        }

        return nearest_neighbor

    def add(self, image_id: str, emb: np.ndarray, image_class: str) -> None:
        self.ids.append(image_id)
        self.classes.append(image_class)
        self.index.add(emb)

    def reset(self) -> None:
        self.ids.clear()
        self.classes.clear()
        self.index.reset()
