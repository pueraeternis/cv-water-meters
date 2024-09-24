from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from shapely import Polygon
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


def is_inside_panel(
    digit_coords: tuple[int, int, int, int], panel_coords: tuple[int, int, int, int]
) -> bool:
    """Проверяет, пересекается ли полигон цифры с полигоном панели."""
    digit_poly = Polygon(
        [
            (digit_coords[0], digit_coords[1]),
            (digit_coords[2], digit_coords[1]),
            (digit_coords[2], digit_coords[3]),
            (digit_coords[0], digit_coords[3]),
        ]
    )
    panel_poly = Polygon(
        [
            (panel_coords[0], panel_coords[1]),
            (panel_coords[2], panel_coords[1]),
            (panel_coords[2], panel_coords[3]),
            (panel_coords[0], panel_coords[3]),
        ]
    )
    return digit_poly.intersects(panel_poly)


def build_polygon(coords: list[int]) -> Polygon:
    """Строит полигон по координатам."""
    poly = Polygon(
        [
            (coords[0], coords[1]),
            (coords[2], coords[1]),
            (coords[2], coords[3]),
            (coords[0], coords[3]),
        ]
    )
    return poly


def filter_duplicates_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты на основе x1 координат и уверенности."""

    df.sort_values(by="x1", inplace=True, ignore_index=True)

    idx_for_drop = []
    for i in range(1, len(df)):
        poly1 = build_polygon(df.iloc[i - 1]["xyxy"])
        poly2 = build_polygon(df.iloc[i]["xyxy"])
        intersection = poly1.intersection(poly2)
        overlap = intersection.area / poly1.area

        if overlap > 0.5:
            df_2rows = df.iloc[i - 1 : i + 1]
            min_conf_index = df_2rows["conf"].idxmin()
            idx_for_drop.append(min_conf_index)

    df.drop(idx_for_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def process_digits_results(
    panels: List[DetectedObject], digits: List[DetectedObject]
) -> List[DetectedObject]:
    """Удаляет дубликаты классов цифр, выбирая наиболее вероятные, которые пересекаются с панелями."""

    new_digits = []
    panel_coords = panels[0].xyxy[0]

    for pred_dig in digits:
        # Преобразование в DataFrame
        digits_df = pd.DataFrame(
            {"cls": pred_dig.cls, "conf": pred_dig.conf, "xyxy": pred_dig.xyxy}
        )
        digits_df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(
            digits_df["xyxy"].tolist(), index=digits_df.index
        )

        # Удаление дубликатов
        digits_df = filter_duplicates_by_confidence(digits_df)

        # Проверка, пересекаются ли цифры с панелью
        digits_df["inside_panel"] = digits_df.apply(
            lambda row: is_inside_panel(
                (row["x1"], row["y1"], row["x2"], row["y2"]), panel_coords  # type: ignore
            ),
            axis=1,
        )  # type: ignore

        # Фильтрация по нахождению внутри панели
        filtered_df = digits_df[digits_df["inside_panel"]].copy()

        # Обновление результатов
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
