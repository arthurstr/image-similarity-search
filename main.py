from fastapi import FastAPI, UploadFile
import uvicorn
from src.service.image_service import ImageService

app = FastAPI()
image_service = ImageService()

@app.put("/images")
async def upload_image(image: UploadFile, image_class: str):
    await image_service.upload_image(image, image_class)
    return {"message": "Image saved successfully" }

@app.post("/images/search")
async def search_similar_image(image: UploadFile):
    result = await image_service.search_similar_image(image)
    return {"results": result}

@app.delete("/images")
def delete_images():
    image_service.delete_images()
    return {"message": "All images deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

