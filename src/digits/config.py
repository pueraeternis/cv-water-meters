from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_path: str
    images_data_path: str
    labels_data_path: str
    labels_output_path: str
    class_labels: dict


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


dataset_config = DatasetConfig(
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

train_config = TrainConfig(
    model_path="yolov8n.pt",
    yaml_path="/home/vbabchuk/research/cv-water-meters/data/datasets/digits/data.yaml",
    batch_size=32,
    img_size=640,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/digits/runs/training",
)
