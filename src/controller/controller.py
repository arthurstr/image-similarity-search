import uvicorn
from fastapi import FastAPI, UploadFile
from starlette import status

from src.service.image_similarity_search_service import ImageSimilaritySearchService

app = FastAPI()
image_service = ImageSimilaritySearchService()

@app.put("/images/{image_id}")
def add_image(image: UploadFile, image_class: str) -> int:
    image_service.add_image(image, image_class)
    return status.HTTP_201_CREATED


@app.post("/images/search")
def search_similar_images(image: UploadFile) -> dict:
    result = image_service.search_similar_image(image)
    return result


@app.delete("/images")
def reset_images() -> int:
    image_service.reset_images()
    return status.HTTP_200_OK


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
