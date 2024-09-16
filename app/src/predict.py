import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class DetectedObject:
    image: str
    names: Dict[int, str]
    cls: List[int]
    conf: List[float]
    xyxy: List[List[int]]


def predict(task: str, image: np.ndarray, model_path: Path) -> List[Results]:
    """Возвращает предсказания модели."""
    model = YOLO(model_path, task=task)
    results = model(
        image,
        device=0,
        conf=0.25,
        save=False,
        save_txt=False,
        save_crop=False,
        exist_ok=True,
        show_labels=False,
        show_conf=False,
    )
    return results


def extract_detected_object_from_results(
    results: List[Results],
) -> List[DetectedObject]:
    """Извлекает координаты объектов из результатов предсказания."""
    objects = []
    for result in results:
        image = Path(result.path).name
        names = result.names
        cls = result.boxes.cls.cpu().numpy().astype(int).tolist()  # type: ignore
        conf = result.boxes.conf.cpu().tolist()  # type: ignore
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int).tolist()  # type: ignore

        objects.append(DetectedObject(image, names, cls, conf, xyxy))

    return objects


def process_digits_results(digits: List[DetectedObject]) -> List[DetectedObject]:
    """Удаляет дубли классов, выбирая наиболее вероятные."""

    def calculate_distance(x1, y1, x2, y2, x3, y3):
        """Вычисляет расстояние от точки до прямой."""
        numerator = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator

    new_digits = []
    for pred_dig in digits:
        digits_df = pd.DataFrame(
            {"cls": pred_dig.cls, "conf": pred_dig.conf, "xyxy": pred_dig.xyxy}
        )

        digits_df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(
            digits_df["xyxy"].tolist(), index=digits_df.index
        )
        digits_df["x1_floor"] = digits_df["x1"].round(-1)
        digits_df.sort_values(by=["x1_floor", "conf"], inplace=True, ignore_index=True)
        digits_df.drop_duplicates(
            subset=["x1_floor"], keep="last", inplace=True, ignore_index=True
        )

        x1, y1 = digits_df.iloc[0][["x1", "y1"]]
        x2, y2 = digits_df.iloc[-1][["x1", "y1"]]
        digits_df["distance"] = digits_df.apply(
            lambda row: calculate_distance(x1, y1, x2, y2, row["x1"], row["y1"]), axis=1
        )

        filtered_df = digits_df[digits_df["distance"] < 20].copy()
        filtered_df.reset_index(drop=True, inplace=True)

        pred_dig.cls = filtered_df["cls"].tolist()
        pred_dig.conf = filtered_df["conf"].tolist()
        pred_dig.xyxy = filtered_df["xyxy"].tolist()

        new_digits.append(pred_dig)

    return new_digits


def extract_value(pred: DetectedObject) -> str:
    """Извлекает значение показаний счетчика из предсказания."""
    sorted_digits = sorted(
        zip(pred.cls, [coord[0] for coord in pred.xyxy]), key=lambda x: x[1]
    )
    value = "".join(str(val[0]) for val in sorted_digits)
    return value
