from config import TestConfig, test_config
from ultralytics import YOLO


def test_model(
    task: str,
    config: TestConfig,
) -> None:
    """Определяет объекты на изображении."""

    # Загрузить модель
    model = YOLO(config.model_path, task=task)

    # Получить предсказания
    model(
        config.imgs_path,
        device=0,
        imgsz=config.img_size,
        conf=config.conf,
        save=True,
        save_txt=True,
        save_crop=False,
        project=config.project_path,
        exist_ok=True,
        show_labels=True,
        show_conf=False,
    )


if __name__ == "__main__":
    test_model(task="detect", config=test_config)
