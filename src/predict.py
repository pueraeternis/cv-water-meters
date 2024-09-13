import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class DetectedObject:
    image: str
    names: dict[int, str]
    cls: list[int]
    conf: list[float]
    xyxy: list[int]


def predict(
    task: str, images: list[str], model_path: Path, project_path: Path
) -> list[Results]:
    """Возвращает предсказания модели."""

    # Загрузить модель
    model = YOLO(model_path, task=task)

    # Получить предсказания
    results = model(
        images,
        device=0,
        conf=0.25,
        save=False,
        save_txt=False,
        save_crop=False,
        project=project_path,
        exist_ok=True,
        show_labels=True,
        show_conf=False,
    )

    return results


def extraxt_detected_object_from_results(
    results: list[Results],
) -> list[DetectedObject]:
    """Извлекает координаты панелей из результатов предсказания."""

    objects = []
    for result in results:
        image = Path(result.path).name
        names = result.names
        cls = result.boxes.cls.cpu().numpy().astype("int").tolist()  # type: ignore
        conf = result.boxes.conf.cpu().tolist()  # type: ignore
        xyxy = result.boxes.xyxy.cpu().numpy().astype("int").tolist()  # type: ignore

        objects.append(DetectedObject(image, names, cls, conf, xyxy))

    return objects


def process_digits_results(digits: list[DetectedObject]) -> list[DetectedObject]:
    """Убирает из результатов дубли классов, выбирая наиболее вероятный."""

    new_digits = []
    for pred_dig in digits:
        digits_df = pd.DataFrame(
            {
                "cls": pred_dig.cls,
                "conf": pred_dig.conf,
                "xyxy": pred_dig.xyxy,
            }
        )
        digits_df["x1"] = digits_df["xyxy"].apply(lambda x: x[0])
        digits_df["y1"] = digits_df["xyxy"].apply(lambda y: y[1])
        digits_df["x1_floor"] = digits_df["x1"].round(-1)
        digits_df.sort_values(by=["x1_floor", "conf"], inplace=True, ignore_index=True)
        digits_df.drop_duplicates(
            subset=["x1_floor"], keep="last", inplace=True, ignore_index=True
        )

        x1, y1 = digits_df.iloc[0]["x1"], digits_df.iloc[0]["y1"]
        x2, y2 = digits_df.iloc[-1]["x1"], digits_df.iloc[-1]["y1"]
        digits_df["distance"] = digits_df.apply(
            lambda row: distance_point_to_line(x1, y1, x2, y2, row["x1"], row["y1"]),
            axis=1,
        )

        digits_df = digits_df[digits_df.distance < 20].copy()
        digits_df.reset_index(inplace=True)

        pred_dig.cls = digits_df.cls.tolist()
        pred_dig.conf = digits_df.conf.tolist()
        pred_dig.xyxy = digits_df.xyxy.tolist()

        new_digits.append(pred_dig)

    return new_digits


def distance_point_to_line(x1, y1, x2, y2, x3, y3):
    """Вычисляет расстояние от точки до прямой."""

    numerator = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distance = numerator / denominator
    return distance


def extract_value(pred: DetectedObject) -> dict[str, str]:
    """Вынимает значение показаний счетчика из предсказания."""

    x1 = [coord[0] for coord in pred.xyxy]  # type: ignore
    value = [str(val[0]) for val in sorted(zip(pred.cls, x1), key=lambda x: x[1])]
    value_str = "".join(value)

    return {"image_name": pred.image, "value": value_str}
