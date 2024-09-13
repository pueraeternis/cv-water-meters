from pathlib import Path

import cv2
from cv2.typing import MatLike

from src.predict import DetectedObject

COLORS = {
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
    start_point: tuple[int, int],
    end_point: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    alpha: float = 0.8,
):
    """Наносит прямоугольник меток с прозрачностью."""

    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, thickness)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_panel_rectangle(image: MatLike, coords: list[int]):
    """Наносит панель счетчика."""

    for coord in coords:
        x1, y1, x2, y2 = map(int, coord)  # type: ignore
        image = draw_transparent_rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 7)

    return image


def draw_digits_rectangle(
    image: MatLike, classlabels: dict[int, str], classes: list[int], coords: list[int]
) -> MatLike:
    for cls, coord in zip(classes, coords):
        x1, y1, x2, y2 = map(int, coord)  # type: ignore

        image = draw_transparent_rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        text = classlabels[cls]
        text_color_bg = COLORS[cls]
        text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_w, text_h = text_size

        image = draw_transparent_rectangle(
            image,
            (x1 - 2, y1 - text_h - 40),
            (x1 + text_w + 10, y1 - 10),
            text_color_bg,
            -1,
        )

        cv2.putText(
            image,
            text,
            (x1 + 5, y1 - 26),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


def visualize(
    image_path: Path,
    panels: list[DetectedObject],
    digits: list[DetectedObject],
    output_dir: Path,
) -> None:
    """Наносит результаты модели на изображение с улучшенной визуализацией."""

    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.name
    image = cv2.imread(str(image_path))
    target_panels = [panel for panel in panels if panel.image == image_name][0]
    target_digits = [digit for digit in digits if digit.image == image_name][0]

    image = draw_panel_rectangle(image, target_panels.xyxy)
    image = draw_digits_rectangle(
        image, target_digits.names, target_digits.cls, target_digits.xyxy
    )

    out_path = output_dir / f"pred_{image_name}"
    cv2.imwrite(str(out_path), image)
