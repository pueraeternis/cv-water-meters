from ultralytics import YOLO

from config import TestConfig, test_config_digits, test_config_panels


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


def main():
    test_model(task="segment", config=test_config_panels)
    test_model(task="detect", config=test_config_digits)


if __name__ == "__main__":
    main()
