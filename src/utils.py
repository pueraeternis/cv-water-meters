import yaml


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию для создания датасета и обучения моделей."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
