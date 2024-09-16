import io
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from src.predict import (
    extract_detected_object_from_results,
    extract_value,
    predict,
    process_digits_results,
)
from src.visualize import visualize

# Конфигурация моделей
MODEL_PANELS = Path("models/panels_base.pt")
MODEL_DIGITS = Path("models/digits_base.pt")

app = FastAPI()


def process_image(image: UploadFile) -> np.ndarray:
    """Конвертирует загруженное изображение в массив NumPy."""

    img = Image.open(image.file)
    return np.array(img)


def get_visualized_image(img: np.ndarray) -> io.BytesIO:
    """Возвращает визуализированное изображение как поток байтов."""

    # Найти панели показаний на изображениях счетчиков
    panels_results = predict("segment", img, MODEL_PANELS)
    panels = extract_detected_object_from_results(panels_results)

    # Определить показания
    digits_results = predict("detect", img, MODEL_DIGITS)
    digits = extract_detected_object_from_results(digits_results)
    digits = process_digits_results(digits)

    # Визуализация результатов работы модели
    img_pred = visualize(img, panels, digits)
    img_bytes = io.BytesIO()
    Image.fromarray(img_pred).save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Welcome to the service for recognizing water meter readings!"}


@app.post("/image/visualize")
async def visualize_results(image: UploadFile = File(...)) -> StreamingResponse:
    img = process_image(image)
    img_bytes = get_visualized_image(img)
    return StreamingResponse(img_bytes, media_type="image/jpeg")


@app.post("/image/readings")
async def read_results(image: UploadFile = File(...)) -> Dict[str, str]:
    img = process_image(image)
    digits_results = predict("detect", img, MODEL_DIGITS)
    digits = extract_detected_object_from_results(digits_results)
    digits = process_digits_results(digits)
    value = extract_value(digits[0])
    return {"description": f"Values: {value}"}
