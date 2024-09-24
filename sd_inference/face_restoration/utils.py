from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np

def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    '''
    Performs a dilation operation on an image, i.e expands the white regions in it. In other words, dilation is the inverse of erotion.
    :param image:
    :param value: If value is <= 0 then we skip the dilation
    :return: the dilated image
    '''
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)

def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    '''
    Applies blurring to the image
    :param image:
    :param value:
    :return:
    '''
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)

def bbox_padding(
        bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32
) -> tuple[int, int, int, int]:
    '''
    Adds padding to a bounding box while ensuring it stays within an image boundaries.
    :param bbox: the coords of the bounding box in an image
    :param image_size: the size of the image in pixels
    :param value: Amount of padding to add
    :return: new coordinates for the bounding box after applying the padding
    '''
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size) # ensure new coordinates don't exceed the image boundaries (i.e the image size)

    # Flatten back into a 1D tuple and return
    return tuple(arr.flatten())

def composite(
        init: Image.Image,
        mask: Image.Image,
        gen: Image.Image,
        bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    '''
    Combines multiple images to create a final composite image.
    :param init: the initial background image
    :param mask: the mask of the image
    :param gen: a generated image to be composited into the initial image
    :param bbox_padded: coordinates of where the generated image is going to be placed
    :return:
    '''
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    # Blend the masked initial image with the pasted generated image
    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)

    return output.convert("RGB")