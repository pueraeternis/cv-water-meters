from unittest.mock import mock_open

import cv2
import numpy as np
import pytest

from src.images import crop_image, crop_panel, extract_coordinates


@pytest.fixture
def test_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
    return image


@pytest.fixture
def test_pts():
    return np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)


@pytest.fixture
def mock_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_labels():
    return "30,30 70,70"


@pytest.mark.parametrize(
    "labels, width, height",
    [("0 0.40 0.20 0.60 0.50", 100, 100)],
)
def test_extract_coordinates(labels: str, width: int, height: int):
    pts = extract_coordinates(labels, width, height)
    np.testing.assert_array_equal(pts, np.array([(40, 20), (60, 50)]))


def test_crop_image(test_image, test_pts):
    cropped_image, x, y = crop_image(test_image, test_pts)

    assert cropped_image.shape == (41, 41, 3)
    assert x == 30
    assert y == 30
    assert np.all(cropped_image == 255)


def test_crop_panel(mocker, mock_image, mock_labels):
    mock_imread = mocker.patch("cv2.imread", return_value=mock_image)
    mock_open_file = mocker.patch("builtins.open", mock_open(read_data=mock_labels))
    mock_pts = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)
    mock_extract_coordinates = mocker.patch(
        "src.images.extract_coordinates", return_value=mock_pts
    )
    mock_imwrite = mocker.patch("cv2.imwrite")

    image_path = "test_image.jpg"
    labels_path = "test_labels.txt"
    crop_panel(image_path, labels_path)

    mock_imread.assert_called_once_with(image_path)
    mock_open_file.assert_called_once_with(labels_path)
    mock_extract_coordinates.assert_called_once_with(
        mock_labels, mock_image.shape[1], mock_image.shape[0]
    )

    assert mock_imwrite.call_count == 1
    args, kwargs = mock_imwrite.call_args
    expected_filename = "test_image_topleft_30_30.jpg"
    expected_cropped_image = mock_image[30:71, 30:71]

    assert args[0] == expected_filename
    assert np.array_equal(args[1], expected_cropped_image)
