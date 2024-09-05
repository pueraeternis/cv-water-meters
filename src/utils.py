import yaml


def load_config(config_path: str) -> dict:
    """Load config for creating datasets and training models."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
