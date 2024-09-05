import yaml

from src.datasets import YOLODatasetDigits, YOLODatasetPanels


def load_config(config_path: str) -> dict:
    """Загружает конфиграционный файл."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("config.yaml")
    panels_ds = YOLODatasetPanels(**config["panels"]["dataset"])
    digits_ds = YOLODatasetDigits(**config["digits"]["dataset"])

    panels_ds.create_dataset()
    digits_ds.create_dataset()


if __name__ == "__main__":
    main()
