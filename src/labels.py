import json
import os

import pandas as pd


def extract_labels(labels_data_filepath: str) -> None:
    """Извлекает координаты полигона панели счетчика."""

    data = pd.read_csv(labels_data_filepath)
    for _, row in data.iterrows():
        image_name = row["photo_name"]
        labels_path = os.path.join(os.path.dirname(labels_data_filepath), "labels")
        label_str = extract_label_from_row(row)
        save_label(image_name, label_str, labels_path)


def extract_label_from_row(row: pd.Series) -> str:
    """Извлекает метки из строки датасета."""

    class_label = ["0"]
    location = row["location"]
    panel_coordinates = json.loads(location.replace("'", '"'))["data"]
    coordinates = [
        str(coord) for point in panel_coordinates for coord in point.values()
    ]
    label = class_label + coordinates
    label_str = " ".join(label)
    return label_str


def save_label(image_name: str, label_str: str, labels_path: str) -> None:
    """Сохраняет разметку панели счетчика в текстовый файл."""

    os.makedirs(labels_path, exist_ok=True)

    filename, _ = os.path.splitext(image_name)
    label_name = f"{filename}.txt"

    label_path = os.path.join(labels_path, label_name)
    with open(label_path, "wt", encoding="utf-8") as f:
        f.write(label_str)
