import os

import cv2
import numpy as np
from cv2.typing import MatLike


def extract_coordinates(labels: str, width: int, height: int) -> np.ndarray:
    coords: list[str] = labels[2:].split(" ")
    labels_rel = [float(point) for point in coords]
    pts = np.array(
        [[x * width, y * height] for x, y in zip(labels_rel[::2], labels_rel[1::2])],
        dtype=np.int32,
    )
    return pts


def crop_image(image: MatLike, pts: np.ndarray) -> tuple[np.ndarray, int, int]:
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = image[y : y + h, x : x + w].copy()

    return croped, x, y


def crop_panel(image_path: str, labels_path: str) -> None:
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    with open(labels_path) as f:
        data = f.read()

    pts = extract_coordinates(data, width, height)
    croped_img, x, y = crop_image(img, pts)

    img_name = os.path.basename(image_path)
    im_name, ext = os.path.splitext(img_name)
    output_name = f"{im_name}_topleft_{x}_{y}{ext}"
    cv2.imwrite(output_name, croped_img)


def main():
    img_path = "data/WaterMeters/images/id_1240_value_28_306.jpg"
    lbl_path = "data/WaterMeters/labels/id_1240_value_28_306.txt"
    crop_panel(img_path, lbl_path)


if __name__ == "__main__":
    main()
