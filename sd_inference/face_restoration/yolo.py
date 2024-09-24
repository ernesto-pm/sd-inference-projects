import os.path

from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from sd_inference.misc.file_utils import get_models_path


def create_mask_from_bbox(
        bboxes: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Creates a binary mask from a set of bounding boxes.
    :param bboxes: The coordinate array of bounding boxes
    :param shape: A tuple representing the dimensions of the output masks
    :return: An image where the area inside the bounding box is white and the area outside the box is black
    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill="white")
        masks.append(mask)
    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def yolo_detector(
        image: Image.Image, model_path: str | Path | None = None, confidence: float = 0.3
) -> list[Image.Image] | None:
    '''
    Detects a face in the image and returns a list of images that are the masks of the faces detected
    :param image:
    :param model_path:
    :param confidence:
    :return:
    '''
    if not model_path:
        model_path = os.path.join(get_models_path(), 'face_detection', 'face_yolov8n.pt')
    model = YOLO(model_path)
    pred = model(image, conf=confidence)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return None

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    return masks