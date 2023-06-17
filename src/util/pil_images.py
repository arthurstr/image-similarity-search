from PIL import Image
import io
async def get_pil_images(image):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))

    return image