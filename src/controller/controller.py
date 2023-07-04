import uvicorn
from fastapi import FastAPI, UploadFile
from starlette import status

from src.service.similarity_search_service import ImageSimilaritySearchService

app = FastAPI()
image_service = ImageSimilaritySearchService()

@app.put("/add/{image_id}")
def add(image: UploadFile, image_class: str) -> int:
    image_service.add(image, image_class)
    return status.HTTP_201_CREATED


@app.post("/search")
def search(image: UploadFile) -> dict:
    result = image_service.search(image)
    return result


@app.delete("/reset")
def reset() -> int:
    image_service.reset()
    return status.HTTP_200_OK


