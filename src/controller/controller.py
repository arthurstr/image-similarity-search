from fastapi import FastAPI, UploadFile
import uvicorn
from starlette import status

from src.service.ImageSimilaritySearchService import ImageSimilaritySearchService

app = FastAPI()
image_service = ImageSimilaritySearchService()

@app.put("/images")
async def upload_image(image: UploadFile, image_class: str):
    await image_service.upload_image(image, image_class)
    return status.HTTP_201_CREATED

@app.post("/images/search")
async def search_similar_image(image: UploadFile) -> dict:
    result = await image_service.search_similar_image(image)
    return {"results": result}

@app.delete("/images")
def delete_images():
    image_service.delete_images()
    return status.HTTP_200_OK

# uvicorn src.controller.controller:app --host 0.0.0.0 --port 8000
#if __name__ == "__main__":
#   uvicorn.run(app, host="0.0.0.0", port=8000)

