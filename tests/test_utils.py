import pytest

from src.utils import load_config


@pytest.fixture(name="yaml_data")
def sample_yaml_data():
    return """
    key1: value1
    key2:
      subkey1: value2
      subkey2: value3
    """


def test_load_config(mocker, yaml_data):
    # Используем mocker, чтобы замокать open
    mock_open = mocker.mock_open(read_data=yaml_data)
    mocker.patch("builtins.open", mock_open)

    config = load_config("dummy_path.yaml")

    assert isinstance(config, dict)
    assert config["key1"] == "value1"
    assert config["key2"]["subkey1"] == "value2"
    assert config["key2"]["subkey2"] == "value3"
