from glob import glob

from ultralytics import YOLO

IMAGES_PATH = "data/images"
MODEL_PATH = "models/inference/digits_base.pt"


def predict(images: list[str]):
    # Загрузить модель
    model = YOLO(MODEL_PATH, task="detect")

    # Получить предсказания
    model(
        images,
        device=0,
        conf=0.25,
        save=True,
        save_txt=True,
        save_crop=False,
        project=IMAGES_PATH,
        exist_ok=True,
        show_labels=True,
        show_conf=False,
    )


def main():
    images = glob(f"{IMAGES_PATH}/*.jpg")
    predict(images)
    print("Images saved in:", IMAGES_PATH)


if __name__ == "__main__":
    main()
