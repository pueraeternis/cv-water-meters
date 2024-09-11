from dataclasses import dataclass
from enum import Enum, auto


class Task(Enum):
    SEGMENT = auto()
    DETECT = auto()


@dataclass
class DatasetConfig:
    dataset_path: str
    images_data_path: str
    labels_data_path: str
    class_labels: dict


@dataclass
class PanelsDatasetConfig(DatasetConfig):
    labels_data_filepath: str


@dataclass
class DigitsDatasetConfig(DatasetConfig):
    labels_output_path: str


@dataclass
class TrainConfig:
    model_path: str
    yaml_path: str
    batch_size: int
    img_size: int
    project_path: str


@dataclass
class TestConfig:
    model_path: str
    imgs_path: str
    img_size: int
    conf: float
    project_path: str


dataset_config_panels = PanelsDatasetConfig(
    dataset_path="/home/vbabchuk/research/cv-water-meters/data/datasets/panels",
    images_data_path="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/images",
    labels_data_path="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/labels",
    labels_data_filepath="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/data.csv",
    class_labels={0: "Panel"},
)

dataset_config_digits = DigitsDatasetConfig(
    dataset_path="/home/vbabchuk/research/cv-water-meters/data/datasets/digits",
    images_data_path="/data1/jupiter/label-data/datasets/water-meters/images",
    labels_data_path="/data1/jupiter/label-data/datasets/water-meters/labels",
    labels_output_path="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/label_studio",
    class_labels={
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    },
)

train_config_panels = TrainConfig(
    model_path="yolov8n-seg.pt",
    yaml_path="/home/vbabchuk/research/cv-water-meters/data/datasets/panels/data.yaml",
    batch_size=32,
    img_size=640,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/panels/runs/training",
)


train_config_digits = TrainConfig(
    model_path="yolov8n.pt",
    yaml_path="/home/vbabchuk/research/cv-water-meters/data/datasets/digits/data.yaml",
    batch_size=32,
    img_size=640,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/digits/runs/training",
)


test_config_panels = TestConfig(
    model_path="/home/vbabchuk/research/cv-water-meters/models/inference/panels_base.pt",
    imgs_path="/home/vbabchuk/research/cv-water-meters/data/datasets/panels/test/images",
    img_size=640,
    conf=0.25,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/panels/runs/predicting",
)


test_config_digits = TestConfig(
    model_path="/home/vbabchuk/research/cv-water-meters/models/inference/digits_base.pt",
    imgs_path="/home/vbabchuk/research/cv-water-meters/data/datasets/digits/test/images",
    img_size=640,
    conf=0.25,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/digits/runs/predicting",
)
