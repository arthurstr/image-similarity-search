import torch
from transformers import AutoImageProcessor, AutoModel
from fastapi import FastAPI, UploadFile
from PIL import Image
import faiss
import uvicorn
import io

processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
model.eval()

index = faiss.IndexFlatL2(1024)

database = {}

app = FastAPI()


@app.put("/images")
async def upload_image(image: UploadFile, image_class: str):
    contents = await image.read()
    image_id = image.filename
    image = Image.open(io.BytesIO(contents))
    inputs = processor(image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.pooler_output.numpy()

    database[image_id] = {
        #"vector": emb.tolist(),  # Convert NumPy array to list
        "class": image_class
    }

    # Add vector to the index
    index.add(emb)

    return {"message": "Image saved successfully" }


@app.post("/images/search")
async def search_similar_image(image: UploadFile):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))
    inputs = processor(image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.pooler_output.numpy()

    D, I = index.search(emb, k=1)

    nearest_neighbor = {}
    nearest_neighbor["id"] = list(database.keys())[I[0][0]]
    nearest_neighbor["class"] = database[nearest_neighbor["id"]]["class"]
    nearest_neighbor["distance"] = float(D[0][0])

    return {"results": nearest_neighbor}


@app.get("/images")
async def get_images():
    return {"database": database}


@app.delete("/images")
async def delete_images():
    database.clear()
    index.reset()

    return {"message": "All images deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
