import json
import os


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


if __name__ == "__main__":
    labels_data_path = "/data1/jupiter/label-data/datasets/water-meters/labels"
    labels_output_path = (
        "/home/vbabchuk/research/cv-water-meters/data/WaterMeters/label_studio"
    )

    class_labels = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }
    extract_label_studio_labels(labels_data_path, labels_output_path, class_labels)
