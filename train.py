import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

from src.config import (
    Task,
    TrainConfig,
    dataset_config_digits,
    dataset_config_panels,
    train_config_digits,
    train_config_panels,
)
from src.datasets import copy_split_data, create_config_file, train_test_split
from src.labels import extract_label_studio_labels, extract_labels


def train_model(task: Task, config: TrainConfig, epochs: int) -> None:
    """Обучает модель YOLO."""

    torch.cuda.empty_cache()

    # Задача сегментации изображения
    if task == Task.SEGMENT:
        args = {
            "model": config.model_path,
            "data": config.yaml_path,
            "epochs": epochs,
            "batch": config.batch_size,
            "imgsz": config.img_size,
            "save": True,
            "project": config.project_path,
            "device": 0,
            "plots": True,
        }
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()

    # Задача определения объектов
    elif task == Task.DETECT:
        model = YOLO(config.model_path)
        model.train(
            data=config.yaml_path,
            epochs=epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            save=True,
            project=config.project_path,
            device=0,
            plots=True,
        )


def train_panels(epochs: int = 1):
    """Обучает модель определения панели показаний."""

    # Конвертация меток в формат YOLO
    extract_labels(dataset_config_panels.labels_data_filepath)

    # Разделение датасета на обучающий, проверочный и тестовый
    datasets = train_test_split(
        image_dir=dataset_config_panels.images_data_path,
        label_dir=dataset_config_panels.labels_data_path,
    )

    # Копирование изображений и меток в train, val, test
    for dir_name, path_list in zip(["train", "val", "test"], datasets):
        copy_split_data(
            dir_name,
            path_list,
            dataset_config_panels.dataset_path,
        )

    # Создание конфига для обучения модели
    create_config_file(
        dataset_path=dataset_config_panels.dataset_path,
        class_labels=dataset_config_panels.class_labels,
        yaml_path=train_config_panels.yaml_path,
    )

    # Обучение модели
    train_model(Task.SEGMENT, train_config_panels, epochs)


def train_digits(epochs: int = 1):
    """Обучает модель определения значений показаний."""

    # Конвертация меток в формат YOLO
    extract_label_studio_labels(
        dataset_config_digits.labels_data_path,
        dataset_config_digits.labels_output_path,
        dataset_config_digits.class_labels,
    )

    # Разделение датасета на обучающий, проверочный и тестовый
    datasets = train_test_split(
        image_dir=dataset_config_digits.images_data_path,
        label_dir=dataset_config_digits.labels_output_path,
    )

    # Копирование изображений и меток в train, val, test
    for dir_name, path_list in zip(["train", "val", "test"], datasets):
        copy_split_data(
            dir_name,
            path_list,
            dataset_config_digits.dataset_path,
        )

    # Создание конфига для обучения модели
    create_config_file(
        dataset_path=dataset_config_digits.dataset_path,
        class_labels=dataset_config_digits.class_labels,
        yaml_path=train_config_digits.yaml_path,
    )

    # Обучение модели
    train_model(Task.DETECT, train_config_digits, epochs)


if __name__ == "__main__":
    train_panels()
