from unittest import mock

import pytest

from src.utils import load_config


@pytest.fixture
def sample_yaml_data():
    return """
    key1: value1
    key2:
      subkey1: value2
      subkey2: value3
    """


def test_load_config(sample_yaml_data):
    # Мокируем open, чтобы он возвращал sample_yaml_data как содержимое файла
    with mock.patch("builtins.open", mock.mock_open(read_data=sample_yaml_data)):
        config = load_config(
            "dummy_path.yaml"
        )  # Пусть путь будет фиктивным, т.к. мы мокируем open

        # Проверяем, что данные загружены корректно
        assert isinstance(config, dict)
        assert config["key1"] == "value1"
        assert config["key2"]["subkey1"] == "value2"
        assert config["key2"]["subkey2"] == "value3"
