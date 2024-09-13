import json
from pathlib import Path

import pandas as pd


def extract_labels(labels_data_filepath: Path) -> None:
    """Извлекает координаты полигона панели счетчика."""

    data = pd.read_csv(labels_data_filepath)
    for _, row in data.iterrows():
        image_path = labels_data_filepath.parent / "labels" / Path(row["photo_name"])
        label_str = extract_label_from_row(row)
        save_label(label_str, image_path)


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


def save_label(label_str: str, image_path: Path) -> None:
    """Сохраняет разметку панели счетчика в текстовый файл."""

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path = image_path.with_suffix(".txt")
    label_path.write_text(label_str, encoding="utf-8")


def extract_label_studio_labels(
    labels_data_path: Path, labels_output_path: Path, class_labels: dict[int, str]
) -> None:
    """Создает датасет для обучения YOLO."""

    for labels_path in labels_data_path.glob("*.txt"):
        with labels_path.open(encoding="utf-8") as f:
            data = json.load(f)

        labels_str = convert_labels(data, class_labels)
        file_name = Path(data["task"]["data"]["image"])

        labels_output_path.mkdir(parents=True, exist_ok=True)
        output = labels_output_path / file_name.with_suffix(".txt")
        with output.open(encoding="utf-8") as f:
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
