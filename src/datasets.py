import os
import random
import shutil

import yaml


def train_test_split(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    # Установим сид для воспроизводимости
    random.seed(seed)

    # Проверяем, что сумма долей равна 1
    if sum(split_ratios) != 1:
        raise ValueError("The sum of the shares must be equal to 1")

    # Получаем список всех изображений и меток
    image_files = sorted(
        [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    )
    label_files = sorted(
        [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    )

    # Убедимся, что количество изображений и меток совпадает
    if len(image_files) != len(label_files):
        raise ValueError("The number of images and labels must match")

    # Соединяем изображения и метки в пары
    paired_files = list(zip(image_files, label_files))

    # Перемешиваем пары
    random.shuffle(paired_files)

    # Разделяем на train, val и test
    total = len(paired_files)
    train_size = int(split_ratios[0] * total)
    val_size = int(split_ratios[1] * total)

    train_files = paired_files[:train_size]
    val_files = paired_files[train_size : train_size + val_size]
    test_files = paired_files[train_size + val_size :]

    # Функция для создания директорий и копирования файлов
    def create_split_dir(split_name, files):
        split_image_dir = os.path.join(output_dir, split_name, "images")
        split_label_dir = os.path.join(output_dir, split_name, "labels")

        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)

        for img_file, lbl_file in files:
            shutil.copy(os.path.join(image_dir, img_file), split_image_dir)
            shutil.copy(os.path.join(label_dir, lbl_file), split_label_dir)

    # Создаем директории и копируем файлы
    create_split_dir("train", train_files)
    create_split_dir("val", val_files)
    create_split_dir("test", test_files)


def create_config_file(
    dataset_path: str, class_labels: dict[int, str], yaml_path: str
) -> None:
    """Создает YAML-файл для обучения YOLO."""

    data = {
        "path": dataset_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": class_labels,
    }

    try:
        # Гарантирует, что список "names" появится после подкаталогов
        names_data = data.pop("names")

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
