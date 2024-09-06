import os
import tempfile

import pytest

from src.extract_labels import extract_labels, save_label


@pytest.fixture(name="sample_csv_data")
def create_sample_csv_data():

    return """photo_name,value,location
id_53_value_595_825.jpg,595.825,"{'type': 'polygon', 'data': [{'x': 0.30788, 'y': 0.30207}, {'x': 0.30676, 'y': 0.32731}, {'x': 0.53501, 'y': 0.33068}, {'x': 0.53445, 'y': 0.33699}, {'x': 0.56529, 'y': 0.33741}, {'x': 0.56697, 'y': 0.29786}, {'x': 0.53501, 'y': 0.29786}, {'x': 0.53445, 'y': 0.30417}]}"
id_553_value_65_475.jpg,65.475,"{'type': 'polygon', 'data': [{'x': 0.26133, 'y': 0.24071}, {'x': 0.31405, 'y': 0.23473}, {'x': 0.31741, 'y': 0.26688}, {'x': 0.30676, 'y': 0.26763}, {'x': 0.33985, 'y': 0.60851}, {'x': 0.29386, 'y': 0.61449}]}"
"""


@pytest.fixture(name="temp_directory")
def create_temp_directory():

    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_save_label(temp_directory):

    image_name = "test_image.jpg"
    label_str = "0 0.30788 0.30207 0.30676 0.32731"
    labels_path = temp_directory

    save_label(image_name, label_str, labels_path)

    expected_file = os.path.join(labels_path, "test_image.txt")
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert content == label_str


def test_extract_labels(mocker, sample_csv_data, temp_directory):

    labels_data_filepath = os.path.join(temp_directory, "labels.csv")

    # Создаем тестовый CSV файл
    with open(labels_data_filepath, "w", encoding="utf-8") as f:
        f.write(sample_csv_data)

    # Мокаем функцию save_label
    mock_save_label = mocker.patch("src.extract_labels.save_label")

    extract_labels(labels_data_filepath)

    # Проверяем, что save_label была вызвана дважды
    assert mock_save_label.call_count == 2

    # Проверяем вызовы с ожидаемыми значениями
    mock_save_label.assert_any_call(
        "id_53_value_595_825.jpg",
        "0 0.30788 0.30207 0.30676 0.32731 0.53501 0.33068 0.53445 0.33699 0.56529 0.33741 0.56697 0.29786 0.53501 0.29786 0.53445 0.30417",
        os.path.join(temp_directory, "labels"),
    )
    mock_save_label.assert_any_call(
        "id_553_value_65_475.jpg",
        "0 0.26133 0.24071 0.31405 0.23473 0.31741 0.26688 0.30676 0.26763 0.33985 0.60851 0.29386 0.61449",
        os.path.join(temp_directory, "labels"),
    )
