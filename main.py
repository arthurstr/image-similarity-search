from fastapi import FastAPI, UploadFile
import uvicorn
from src.service.image_embedding_service import ImageEmbeddingService
from src.service.similarity_search_service import SimilaritySearchService
from src.util.pil_images import get_pil_images

app = FastAPI()
embedding_service = ImageEmbeddingService()
similarity_search_service = SimilaritySearchService(embedding_service)

@app.put("/images")
async def upload_image(image: UploadFile, image_class: str):
    image_id = image.filename
    image_emb = embedding_service.process_image(await get_pil_images(image))
    similarity_search_service.add_image(image_id, image_emb, image_class)

    return {"message": "Image saved successfully" }

@app.post("/images/search")
async def search_similar_image(image: UploadFile):
    result = similarity_search_service.search_similar_image(await get_pil_images(image))

    return {"results": result}

@app.delete("/images")
def delete_images():
    similarity_search_service.delete_images()

    return {"message": "All images deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

