from PIL import Image
import io
async def decode_image(image):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))

    return image