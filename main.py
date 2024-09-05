from src.datasets import YOLODatasetDigits, YOLODatasetPanels
from src.utils import load_config


def main():
    config = load_config("config.yaml")
    panels_ds = YOLODatasetPanels(**config["panels"]["dataset"])
    digits_ds = YOLODatasetDigits(**config["digits"]["dataset"])

    panels_ds.create_dataset()
    digits_ds.create_dataset()


if __name__ == "__main__":
    main()
