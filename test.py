from ultralytics import YOLO

from src.config import Task, TestConfig, test_config_digits, test_config_panels


def test_model(
    task: Task,
    config: TestConfig,
) -> None:
    """Определяет объекты на изображении."""

    # Загрузить модель
    model = YOLO(config.model_path, task=task.value)

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
    test_model(Task.SEGMENT, config=test_config_panels)
    test_model(Task.DETECT, config=test_config_digits)


if __name__ == "__main__":
    main()
