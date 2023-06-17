from fastapi import FastAPI, UploadFile
from PIL import Image
import uvicorn
import io
from src.service.image_embedding_service import ImageEmbeddingService
from src.service.similarity_search_service import SimilaritySearchService

app = FastAPI()
embedding_service = ImageEmbeddingService()
similarity_search_service = SimilaritySearchService(embedding_service)

@app.put("/images")
async def upload_image(image: UploadFile, image_class: str):
    contents = await image.read()
    image_id = image.filename
    image = Image.open(io.BytesIO(contents))
    image_emb = embedding_service.process_image(image)
    embedding_service.add_image(image_id, image_emb, image_class)
    similarity_search_service.add_index(image_emb)

    return {"message": "Image saved successfully" }

@app.post("/images/search")
async def search_similar_image(image: UploadFile):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))
    result = similarity_search_service.search_similar_image(image)

    return {"results": result}

@app.delete("/images")
async def delete_images():
    embedding_service.delete_images()

    return {"message": "All images deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

