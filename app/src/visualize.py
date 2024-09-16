from typing import Dict, List, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

from src.predict import DetectedObject

# Цвета для визуализации
COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128),
    7: (0, 128, 255),
    8: (128, 128, 0),
    9: (0, 128, 128),
}

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)


def draw_transparent_rectangle(
    image: MatLike,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float = 0.8,
) -> MatLike:
    """Наносит прямоугольник с прозрачностью."""

    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, thickness)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_panel_rectangle(
    image: MatLike, coords: List[Tuple[int, int, int, int]]
) -> MatLike:
    """Наносит панели счетчика."""

    for coord in coords:
        x1, y1, x2, y2 = map(int, coord)
        image = draw_transparent_rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 7)
    return image


def draw_digits_rectangle(
    image: MatLike,
    classlabels: Dict[int, str],
    classes: List[int],
    coords: List[Tuple[int, int, int, int]],
) -> MatLike:
    """Наносит прямоугольники и текст для цифр."""

    for cls, coord in zip(classes, coords):
        x1, y1, x2, y2 = map(int, coord)
        image = draw_transparent_rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        text = classlabels[cls]
        text_color_bg = COLORS.get(cls, (0, 0, 0))  # Цвет по умолчанию
        text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_w, text_h = text_size

        # Рисуем фон для текста
        image = draw_transparent_rectangle(
            image,
            (x1 - 2, y1 - text_h - 40),
            (x1 + text_w + 10, y1 - 10),
            text_color_bg,
            -1,
        )

        # Рисуем текст
        cv2.putText(
            image, text, (x1 + 5, y1 - 26), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS
        )

    return image


def visualize(
    image: np.ndarray, panels: List[DetectedObject], digits: List[DetectedObject]
) -> MatLike:
    """Наносит результаты модели на изображение с визуализацией панелей и цифр."""

    if panels and digits:
        target_panels = panels[0]
        target_digits = digits[0]

        image = draw_panel_rectangle(image, target_panels.xyxy)  # type: ignore
        image = draw_digits_rectangle(
            image, target_digits.names, target_digits.cls, target_digits.xyxy  # type: ignore
        )

    return image
