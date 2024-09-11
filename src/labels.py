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


def extract_label_studio_labels(
    labels_data_path: str, labels_output_path: str, class_labels: dict[int, str]
) -> None:
    """Создает датасет для обучения YOLO."""

    for labels in os.listdir(labels_data_path):
        labels_path = os.path.join(labels_data_path, labels)
        with open(labels_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        labels_str = convert_labels(data, class_labels)
        basename = os.path.basename(data["task"]["data"]["image"])
        file_name = os.path.splitext(basename)[0]

        os.makedirs(labels_output_path, exist_ok=True)
        filepath = os.path.join(labels_output_path, f"{file_name}.txt")
        with open(filepath, "wt", encoding="utf-8") as f:
            for l_str in labels_str:
                f.write(f"{l_str}\n")


def convert_labels(data: dict, class_labels: dict[int, str]) -> list[str]:
    """Конвертирует метки в формат YOLO."""

    labels_str = []

    for label in data["result"]:
        labelclass = int(label["value"]["rectanglelabels"][0])
        x = label["value"]["x"] / 100
        y = label["value"]["y"] / 100
        width = label["value"]["width"] / 100
        height = label["value"]["height"] / 100

        points = [x + (width / 2), y + (height / 2), width, height]

        points = [str(p) for p in points]
        label_str = " ".join([class_labels[labelclass]] + points)
        labels_str.append(label_str)

    return labels_str
