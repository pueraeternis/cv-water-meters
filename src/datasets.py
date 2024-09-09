import os
import random
import shutil

import yaml


def is_file(directory: str, filename: str) -> bool:
    """Проверяет, является ли объект по заданному пути файлом."""

    return os.path.isfile(os.path.join(directory, filename))


def copy_split_data(
    split_name: str,
    files: list[tuple[str, str]],
    image_dir: str,
    label_dir: str,
    output_dir: str,
):
    """Копирует изображения и метки в train, val, test."""

    split_image_dir = os.path.join(output_dir, split_name, "images")
    split_label_dir = os.path.join(output_dir, split_name, "labels")

    os.makedirs(split_image_dir, exist_ok=True)
    os.makedirs(split_label_dir, exist_ok=True)

    for img_file, lbl_file in files:
        shutil.copy(os.path.join(image_dir, img_file), split_image_dir)
        shutil.copy(os.path.join(label_dir, lbl_file), split_label_dir)


def train_test_split(
    image_dir: str,
    label_dir: str,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Разделяет набор данных на train, val, test."""

    if sum(split_ratios) != 1:
        raise ValueError("The sum of the shares must be equal to 1")

    image_files = sorted([f for f in os.listdir(image_dir) if is_file(image_dir, f)])
    label_files = sorted([f for f in os.listdir(label_dir) if is_file(label_dir, f)])

    if len(image_files) != len(label_files):
        raise ValueError("The number of images and labels must match")

    paired_files = list(zip(image_files, label_files))
    random.seed(seed)
    random.shuffle(paired_files)

    total = len(paired_files)
    train_size = int(split_ratios[0] * total)
    val_size = int(split_ratios[1] * total)

    train_files = paired_files[:train_size]
    val_files = paired_files[train_size : train_size + val_size]
    test_files = paired_files[train_size + val_size :]

    return train_files, val_files, test_files


def create_config_file(
    dataset_path: str, class_labels: dict[int, str], yaml_path: str
) -> None:
    """Создает YAML-файл для обучения YOLO."""

    data = {
        "path": dataset_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
    }

    names_data = class_labels

    try:
        with open(yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                data,
                yaml_file,
                Dumper=yaml.SafeDumper,
                default_flow_style=False,
                allow_unicode=True,
            )
            yaml_file.write("\n")
            yaml.dump(
                {"names": names_data},
                yaml_file,
                Dumper=yaml.SafeDumper,
                default_flow_style=False,
                allow_unicode=True,
            )

    except IOError as e:
        print(f"Error creating YAML file: {e}")
