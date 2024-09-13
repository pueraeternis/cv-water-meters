import pprint
from pathlib import Path

from src.predict import (
    extract_value,
    extraxt_detected_object_from_results,
    predict,
    process_digits_results,
)
from src.visualize import visualize

IMAGES_PATH = Path("data/images")
RESULTS_PATH = IMAGES_PATH / "results"
MODEL_PANELS = Path("models/inference/panels_base.pt")
MODEL_DIGITS = Path("models/inference/digits_base.pt")


def main():
    # Прочитать изображения из папки
    images = IMAGES_PATH.glob("*.jpg")
    images_str = [str(path) for path in images]

    # Найти панели показаний на изображениях счетчиков
    panels_results = predict("segment", images_str, MODEL_PANELS, IMAGES_PATH)
    panels = extraxt_detected_object_from_results(panels_results)

    # Определить показания
    digits_results = predict("detect", images_str, MODEL_DIGITS, IMAGES_PATH)
    digits = extraxt_detected_object_from_results(digits_results)

    digits = process_digits_results(digits)

    # Визуализация результатов работы модели
    for img in images:
        visualize(img, panels, digits, RESULTS_PATH)

    # Вывод на экран показаний
    values = []
    for pred in digits:
        val = extract_value(pred)
        values.append(val)
    pprint.pprint(values)


if __name__ == "__main__":
    main()
