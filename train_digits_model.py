import torch
from ultralytics import YOLO

from src.datasets import copy_split_data, create_config_file, train_test_split
from src.digits.config import TrainConfig, dataset_config, train_config
from src.digits.labels import extract_label_studio_labels


def train_model(config: TrainConfig, epochs: int = 1) -> None:
    """Обучает модель YOLO."""

    torch.cuda.empty_cache()

    # Задача определения объектов
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


def main():
    # Конвертация меток в формат YOLO
    extract_label_studio_labels(
        dataset_config.labels_data_path,
        dataset_config.labels_output_path,
        dataset_config.class_labels,
    )

    # Разделение датасета на обучающий, проверочный и тестовый
    datasets = train_test_split(
        image_dir=dataset_config.images_data_path,
        label_dir=dataset_config.labels_output_path,
    )

    # Копирование изображений и меток в train, val, test
    for dir_name, dataset_name in zip(["train", "val", "test"], datasets):
        copy_split_data(
            dir_name,
            dataset_name,
            dataset_config.images_data_path,
            dataset_config.labels_output_path,
            dataset_config.dataset_path,
        )

    # Создание конфига для обучения модели
    create_config_file(
        dataset_path=dataset_config.dataset_path,
        class_labels=dataset_config.class_labels,
        yaml_path=train_config.yaml_path,
    )

    # Обучение модели
    train_model(train_config)


if __name__ == "__main__":
    main()
