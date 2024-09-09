import cv2
import numpy as np

img = cv2.imread("data/WaterMeters/images/id_1243_value_13_474.jpg")
height, width, _ = img.shape

labels_path = "data/WaterMeters/labels/id_1243_value_13_474.txt"
with open(labels_path) as f:
    data = f.read()

labels = [float(point) for point in data[2:].split(" ")]

pts = np.array(
    [[x * width, y * height] for x, y in zip(labels[::2], labels[1::2])], dtype=np.int32
)


## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x, y, w, h = rect
croped = img[y : y + h, x : x + w].copy()

## (2) make mask
pts = pts - pts.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
bg = np.ones_like(croped, np.uint8) * 255
cv2.bitwise_not(bg, bg, mask=mask)
dst2 = bg + dst


cv2.imwrite("croped.png", croped)
cv2.imwrite("mask.png", mask)
cv2.imwrite("dst.png", dst)
cv2.imwrite("dst2.png", dst2)
