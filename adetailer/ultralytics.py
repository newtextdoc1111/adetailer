from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
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
    use_bbox_mask: bool = False,
) -> PredictOutput[float]:
    from ultralytics import YOLO, YOLOWorld

    model = YOLO(model_path)
    if isinstance(model, YOLOWorld):
        apply_classes(model, model_path, classes)
        pred = model(image, conf=confidence, device=device)
    else:  # YOLO model
        target_class_ids = get_class_indices(model, classes)
        pred = model.predict(
            image,
            conf=confidence,
            classes=target_class_ids if len(target_class_ids) > 0 else None,
            device=device,
        )

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None or use_bbox_mask:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    # Extract class names
    if pred[0].boxes.cls is not None and len(pred[0].boxes.cls) > 0:
        class_ids = pred[0].boxes.cls.cpu().numpy().astype(int).tolist()
        class_names = [model.names[class_id] for class_id in class_ids]
    else:
        class_names = []

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(
        bboxes=bboxes,
        masks=masks,
        confidences=confidences,
        preview=preview,
        class_names=class_names,
    )


def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def get_class_indices(model: YOLO, classes: str) -> list[int]:
    """
    Get class indices from the model based on the provided class names.
    """
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    return [idx for idx, name in enumerate(model.names.values()) if name in parsed]


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
