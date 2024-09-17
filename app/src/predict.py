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


def process_digits_results(
    panels: List[DetectedObject], digits: List[DetectedObject]
) -> List[DetectedObject]:
    """Удаляет дубли классов, выбирая наиболее вероятные."""

    def is_inside_panel(x1: int, y1: int, x2: int, y2: int) -> bool:
        """Проверяет, находится ли цифра внутри панели."""

        # Получить центр цифры и координаты панели
        center = x1 + (x2 - x1), y1 + (y2 - y1)
        panel_x1, panel_y1, panel_x2, panel_y2 = panels[0].xyxy[0]
        # Проверить, является ли цифра внутри панели
        if (panel_x1 < center[0] < panel_x2) and (panel_y1 < center[1] < panel_y2):
            return True
        return False

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

        digits_df["inside_panel"] = digits_df.apply(
            lambda row: is_inside_panel(row["x1"], row["y1"], row["x2"], row["y2"]),
            axis=1,
        )

        filtered_df = digits_df[digits_df["inside_panel"]].copy()
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
