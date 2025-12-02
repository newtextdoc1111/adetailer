from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Convert segmentation masks to PIL Images.

    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.uint8 or torch.float32, shape=(N, H, W).
        Segmentation masks from Ultralytics YOLO model.
        The device can be CUDA.

    shape: tuple[int, int]
        (W, H) of the original image for resizing.

    Returns
    -------
    list[Image.Image]
        List of PIL Images in mode "L" (grayscale), resized to the target shape.

    Notes
    -----
    This function handles both uint8 (0-1 values) and float32 (0.0-1.0 values) masks
    from different versions of Ultralytics. The masks are converted to 0-255 range
    for proper visualization.
    """
    # Convert to 0-255 range for proper visualization
    # Handles both uint8 (0-1) and float32 (0.0-1.0)
    masks_uint8 = (masks.cpu().numpy() * 255).astype(np.uint8)

    return [to_pil_image(mask, mode="L").resize(shape) for mask in masks_uint8]