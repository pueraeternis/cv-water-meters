import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from src.predict import (
    DetectedObject,
    extract_detected_object_from_results,
    extract_value,
    predict,
    process_digits_results,
)
from src.visualize import visualize

# Конфигурация моделей
MODEL_PANELS = Path("models/panels_base.pt")
MODEL_DIGITS = Path("models/digits_base.pt")

router = APIRouter(prefix="/image", tags=["Predict & Visualize"])


def image_to_array(image: UploadFile) -> np.ndarray:
    """Конвертирует загруженное изображение в массив NumPy."""

    img = Image.open(image.file)
    return np.array(img)


def get_predictions(
    img: np.ndarray,
) -> Tuple[List[DetectedObject], List[DetectedObject]]:
    """Получает предсказания по панели и показаниям счетчиков."""

    # Найти панели показаний на изображениях счетчиков
    panels_results = predict("segment", img, MODEL_PANELS)
    panels = extract_detected_object_from_results(panels_results)

    # Определить показания
    digits_results = predict("detect", img, MODEL_DIGITS)
    digits = extract_detected_object_from_results(digits_results)

    # Если есть результаты, обработать их
    if digits[0].cls:
        digits = process_digits_results(panels, digits)

    return panels, digits


def get_visualized_image(img: np.ndarray) -> io.BytesIO:
    """Возвращает визуализированное изображение как поток байтов."""

    panels, digits = get_predictions(img)
    # Визуализация результатов работы модели
    img_pred = visualize(img, panels, digits)
    img_bytes = io.BytesIO()
    Image.fromarray(img_pred).save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@router.post("/visualize")
async def visualize_results(image: UploadFile = File(...)) -> StreamingResponse:
    img = image_to_array(image)
    img_bytes = get_visualized_image(img)
    return StreamingResponse(img_bytes, media_type="image/jpeg")


@router.post("/readings")
async def read_results(image: UploadFile = File(...)) -> Dict[str, str]:
    img = image_to_array(image)
    _, digits = get_predictions(img)
    value = extract_value(digits[0])
    return {"value": f"{value}"}
