import json
import os
import random
import shutil
from abc import ABC, abstractmethod

import pandas as pd
import yaml


class BaseYOLODataset(ABC):
    """Создание датасета в формате YOLO."""

    def __init__(
        self,
        dataset_path: str,
        images_data_path: str,
        labels_data_path: str,
        class_labels: dict[int, str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ):

        self.dataset_path = dataset_path
        self.images_data_path = images_data_path
        self.labels_data_path = labels_data_path
        self.class_labels = class_labels
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.images_path = os.path.join(dataset_path, "images")
        self.labels_path = os.path.join(dataset_path, "labels")
        self.yaml_file = os.path.join(dataset_path, "data.yaml")

    @abstractmethod
    def create_dataset(self) -> None:
        """Метод для создания датасета."""

    def _create_structure(self) -> None:
        """Создает структуру директорий для датасета."""

        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)

        os.makedirs(self.dataset_path)
        os.makedirs(self.images_path)
        os.makedirs(self.labels_path)

    def _train_val_test_split(self) -> None:
        """Разделяет датасет на тренировочный, валидационный и тестовый."""

        assert self.train_ratio + self.val_ratio + self.test_ratio == 1

        if self.random_state:
            random.seed(self.random_state)

        for set_type in ["train", "val", "test"]:
            if os.path.isdir(os.path.join(self.dataset_path, set_type)):
                shutil.rmtree(os.path.join(self.dataset_path, set_type))
            for content_type in ["images", "labels"]:
                os.makedirs(
                    os.path.join(self.dataset_path, set_type, content_type),
                    exist_ok=True,
                )

        all_files = [
            f
            for f in os.listdir(self.images_path)
            if os.path.isfile(os.path.join(self.images_path, f))
        ]

        random.shuffle(all_files)

        total_files = len(all_files)
        train_end = int(self.train_ratio * total_files)
        val_end = train_end + int(self.val_ratio * total_files)

        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]

        self._copy_files(train_files, "train")
        self._copy_files(val_files, "val")
        self._copy_files(test_files, "test")

        self._remove_directories()

    def _copy_files(self, files: list[str], set_type: str) -> None:
        """Копирует файлы в обучающий, валидационный и тестовый наборы."""

        for file in files:
            shutil.copy(
                os.path.join(self.images_path, file),
                os.path.join(self.dataset_path, set_type, "images"),
            )

            label_file = file.rsplit(".", 1)[0] + ".txt"
            shutil.copy(
                os.path.join(self.labels_path, label_file),
                os.path.join(self.dataset_path, set_type, "labels"),
            )

    def _remove_directories(self) -> None:
        """Удаляет изображения и метки до разделения датасета."""

        dirs = [self.images_path, self.labels_path]
        for d in dirs:
            shutil.rmtree(d)

    def _create_yaml_file(self) -> None:
        """Создает YAML-файл для обучения YOLO."""

        data = {
            "path": self.dataset_path,
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": self.class_labels,
        }

        try:
            # Гарантирует, что список «names» появится после подкаталогов
            names_data = data.pop("names")

            with open(self.yaml_file, "w", encoding="utf-8") as yaml_file:
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


class YOLODatasetPanels(BaseYOLODataset):
    """Подготовка датасета с размеченными панелями показаний счетчиков."""

    def create_dataset(self):
        """Создает датасет для обучения YOLO."""

        self._create_structure()
        self._copy_images()
        self._extract_labels()
        self._train_val_test_split()
        self._create_yaml_file()

        print(f"Dataset successfully created and saved: {self.dataset_path}")

    def _copy_images(self) -> None:
        """Копирует изображения в директорию датасета."""

        shutil.copytree(self.images_data_path, self.images_path, dirs_exist_ok=True)

    def _extract_labels(self) -> None:
        """Извлекает координаты полигона панели счетчика."""

        data = pd.read_csv(self.labels_data_path)
        for _, row in data.iterrows():
            photo_name = row["photo_name"]
            class_label = ["0"]
            location = row["location"]
            panel_coordinates = json.loads(location.replace("'", '"'))["data"]
            coordinates = [
                str(coord) for point in panel_coordinates for coord in point.values()
            ]
            label = class_label + coordinates
            label_str = " ".join(label)

            self._save_label(photo_name, label_str)

    def _save_label(self, image_name: str, label_str: str) -> None:
        """Сохраняет разметку панели счетчика в текстовый файл."""

        filename, _ = os.path.splitext(image_name)
        label_name = f"{filename}.txt"
        label_path = os.path.join(self.labels_path, label_name)
        with open(label_path, "wt", encoding="utf-8") as f:
            f.write(label_str)


class YOLODatasetDigits(BaseYOLODataset):
    """Подготовка датасета с размеченными цифрами показаний счетчиков."""

    def create_dataset(self):
        """Создает датасет для обучения YOLO."""

        self._create_structure()
        self._process_data()
        self._train_val_test_split()
        self._create_yaml_file()

        print(f"Dataset successfully created and saved: {self.dataset_path}")

    def _process_data(self) -> None:
        """Создает датасет для обучения YOLO."""

        for labels in os.listdir(self.labels_data_path):
            labels_path = os.path.join(self.labels_data_path, labels)
            with open(labels_path, "rt", encoding="utf-8") as f:
                data = json.load(f)

            labels_str = self._convert_labels(data)
            basename = os.path.basename(data["task"]["data"]["image"])
            file_name = os.path.splitext(basename)[0]

            filepath = os.path.join(self.labels_path, f"{file_name}.txt")
            with open(filepath, "wt", encoding="utf-8") as f:
                for l_str in labels_str:
                    f.write(f"{l_str}\n")

            self._copy_image(basename)

    def _copy_image(self, basename: str) -> None:
        """Копирует изображение в директорию датасета."""

        shutil.copyfile(
            os.path.join(self.images_data_path, basename),
            os.path.join(self.images_path, basename),
        )

    def _convert_labels(self, data: dict) -> list[str]:
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
            label_str = " ".join([self.class_labels[labelclass]] + points)
            labels_str.append(label_str)

        return labels_str
