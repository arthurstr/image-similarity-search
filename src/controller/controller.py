from fastapi import FastAPI, UploadFile
import uvicorn
from starlette import status

from src.service.image_similarity_search_service import ImageSimilaritySearchService

app = FastAPI()
image_service = ImageSimilaritySearchService()

@app.put("/images")
def upload_image(image: UploadFile, image_class: str) -> int:
    image_service.upload_image(image, image_class)
    return status.HTTP_201_CREATED

@app.post("/images/search")
def search_similar_image(image: UploadFile) -> dict:
    result = image_service.search_similar_image(image)
    return result

@app.delete("/images")
def delete_images() -> int:
    image_service.delete_images()
    return status.HTTP_200_OK


