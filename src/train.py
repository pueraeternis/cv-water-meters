import torch
from ultralytics.models.yolo.segment import SegmentationTrainer

from utils import load_config


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


if __name__ == "__main__":
    config = load_config("config.yaml")
    train_model(**config["panels"]["train"])
