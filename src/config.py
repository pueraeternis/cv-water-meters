from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_path: str
    images_data_path: str
    labels_data_path: str
    labels_data_filepath: str
    class_labels: dict


@dataclass
class TrainConfig:
    model_path: str
    yaml_path: str
    batch_size: int
    img_size: int
    project_path: str


dataset_config = DatasetConfig(
    dataset_path="/home/vbabchuk/research/cv-water-meters/data/datasets/panels",
    images_data_path="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/images",
    labels_data_path="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/labels",
    labels_data_filepath="/home/vbabchuk/research/cv-water-meters/data/WaterMeters/data.csv",
    class_labels={0: "Panel"},
)

train_config = TrainConfig(
    model_path="yolov8n-seg.pt",
    yaml_path="/home/vbabchuk/research/cv-water-meters/data/datasets/panels/data.yaml",
    batch_size=32,
    img_size=640,
    project_path="/home/vbabchuk/research/cv-water-meters/models/train/panels/runs/training",
)
