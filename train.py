import torch
from ultralytics.models.yolo.segment import SegmentationTrainer

from src.datasets import create_config_file, train_test_split
from src.labels import extract_labels
from src.utils import load_config


def train_model(
    model_path: str,
    yaml_path: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    project_path: str,
) -> None:
    """Обучает модель YOLO."""

    torch.cuda.empty_cache()

    # Задача сегментации изображения
    args = {
        "model": model_path,
        "data": yaml_path,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "save": True,
        "project": project_path,
        "device": 0,
        "plots": True,
    }

    trainer = SegmentationTrainer(overrides=args)
    trainer.train()


def main():

    # Загрузка конфигурации
    config = load_config("config.yaml")

    # Конвертация меток в формат YOLO
    extract_labels(config["panels"]["dataset"]["labels_data_filepath"])

    # Разделение датасета на обучающий, проверочный и тестовый
    train_test_split(
        image_dir=config["panels"]["dataset"]["images_data_path"],
        label_dir=config["panels"]["dataset"]["labels_data_path"],
        output_dir=config["panels"]["dataset"]["dataset_path"],
    )

    # Создание конфига для обучения модели
    create_config_file(
        dataset_path=config["panels"]["dataset"]["dataset_path"],
        class_labels=config["panels"]["dataset"]["class_labels"],
        yaml_path=config["panels"]["train"]["yaml_path"],
    )

    # Обучение модели
    train_model(**config["panels"]["train"])


if __name__ == "__main__":
    main()
