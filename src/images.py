import os

import cv2
import numpy as np
from cv2.typing import MatLike


def extract_image_name(image_path: str) -> tuple[str, str]:
    """Извлекет из пути имя файла и расширение."""

    img_name = os.path.basename(image_path)
    im_name, ext = os.path.splitext(img_name)
    return im_name, ext


def extract_coordinates(labels: str, width: int, height: int) -> np.ndarray:
    """Извлекает полигон точек из меток."""

    coords: list[str] = labels[2:].split(" ")
    labels_rel = [float(point) for point in coords]
    pts = np.array(
        [[x * width, y * height] for x, y in zip(labels_rel[::2], labels_rel[1::2])],
        dtype=np.int32,
    )
    return pts


def crop_image(image: MatLike, pts: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Кадрирует изображение, вырезая полигон."""

    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = image[y : y + h, x : x + w].copy()

    return croped, x, y


def crop_panel(image_path: str, labels_path: str) -> None:
    """Вырезает панель счетчика из изображения."""

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    with open(labels_path) as f:
        data = f.read()

    pts = extract_coordinates(data, width, height)
    croped_img, x, y = crop_image(img, pts)

    im_name, ext = extract_image_name(image_path)
    output_name = f"{im_name}_topleft_{x}_{y}{ext}"
    cv2.imwrite(output_name, croped_img)


def image_to_gray_scale(image_path: str) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    im_name, ext = extract_image_name(image_path)
    output_name = f"{im_name}_gray{ext}"
    cv2.imwrite(output_name, img)


def apply_thresh_to_image(image_path: str, thresh_inv: bool = False) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    thresh = cv2.THRESH_BINARY
    if thresh_inv:
        thresh = cv2.THRESH_BINARY_INV

    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh, 11, 2
    )
    im_name, ext = extract_image_name(image_path)
    output_name = f"{im_name}_thresh{ext}"
    cv2.imwrite(output_name, img_thresh)


def main():
    img_path = "data/WaterMeters/images/id_1240_value_28_306.jpg"
    lbl_path = "data/WaterMeters/labels/id_1240_value_28_306.txt"
    img_panel_path = "id_1240_value_28_306_topleft_183_553.jpg"
    crop_panel(img_path, lbl_path)
    image_to_gray_scale(img_panel_path)
    apply_thresh_to_image(img_panel_path)


if __name__ == "__main__":
    main()
