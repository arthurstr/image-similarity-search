from fastapi import UploadFile
from PIL import Image
import io



def decode_image(image: UploadFile) -> Image.Image:
    contents = image.file.read()
    image = Image.open(io.BytesIO(contents))
    return image
