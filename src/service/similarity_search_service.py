import faiss
class SimilaritySearchService:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1024)
        self.ids = []
        self.classes = []

    def search_similar_image(self, image):
        D, I = self.index.search(image, k=1)

        nearest_neighbor = {}
        nearest_neighbor["id"] = self.ids[I[0][0]]
        nearest_neighbor["class"] = self.classes[I[0][0]]
        nearest_neighbor["distance"] = float(D[0][0])

        return nearest_neighbor
    def add_image(self, image_id, emb , image_class):
        self.ids.append(image_id)
        self.classes.append(image_class)
        self.index.add(emb)

    def delete_images(self):
        self.ids.clear()
        self.classes.clear()
        self.index.reset()