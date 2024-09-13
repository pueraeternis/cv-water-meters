import os
import random
import shutil
from pathlib import Path

import yaml


def copy_split_data(
    dir_name: str,
    path_list: list[tuple[Path, Path]],
    output_dir: Path,
):
    """Копирует изображения и метки в train, val, test."""

    split_image_dir = output_dir / dir_name / "images"
    split_label_dir = output_dir / dir_name / "labels"

    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)

    for img_file, lbl_file in path_list:
        shutil.copy(img_file, split_image_dir)
        shutil.copy(lbl_file, split_label_dir)


def train_test_split(
    image_dir: Path,
    label_dir: Path,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Разделяет набор данных на train, val, test."""

    if sum(split_ratios) != 1:
        raise ValueError("The sum of the shares must be equal to 1")

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
    label_files = sorted([p for p in label_dir.iterdir() if p.is_file()])

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
    dataset_path: Path, class_labels: dict[int, str], yaml_path: Path
) -> None:
    """Создает YAML-файл для обучения YOLO."""

    data = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
    }

    names_data = class_labels

    try:
        with yaml_path.open(mode="w", encoding="utf-8") as yaml_file:
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
