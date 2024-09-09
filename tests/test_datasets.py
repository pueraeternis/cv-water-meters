import os

import yaml

from src.datasets import copy_split_data, create_config_file, is_file, train_test_split


def test_is_file(tmpdir):
    test_dir = tmpdir.mkdir("test_dir")
    file_path = test_dir.join("test_file.txt")
    file_path.write("test content")

    assert is_file(str(test_dir), "test_file.txt") is True
    assert is_file(str(test_dir), "non_existing_file.txt") is False


def test_copy_split_data(tmpdir):
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")
    output_dir = tmpdir.mkdir("output")

    img_file = image_dir.join("image1.jpg")
    lbl_file = label_dir.join("label1.txt")
    img_file.write("image content")
    lbl_file.write("label content")

    files = [("image1.jpg", "label1.txt")]

    copy_split_data("train", files, str(image_dir), str(label_dir), str(output_dir))

    assert os.path.exists(output_dir.join("train", "images", "image1.jpg"))
    assert os.path.exists(output_dir.join("train", "labels", "label1.txt"))


def test_train_test_split(tmpdir):
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")

    # Создаем 10 пар изображений и меток
    for i in range(10):
        image_file = image_dir.join(f"image{i}.jpg")
        label_file = label_dir.join(f"label{i}.txt")
        image_file.write(f"image content {i}")
        label_file.write(f"label content {i}")

    train_files, val_files, test_files = train_test_split(
        str(image_dir), str(label_dir)
    )

    assert len(train_files) == 7
    assert len(val_files) == 1
    assert len(test_files) == 2
    assert all(isinstance(pair, tuple) for pair in train_files)
    assert all(isinstance(pair, tuple) for pair in val_files)
    assert all(isinstance(pair, tuple) for pair in test_files)


def test_create_config_file_correct_yaml_dump(mocker):
    # Мокируем функцию open и yaml.dump
    mock_open = mocker.mock_open()
    mocker.patch("builtins.open", mock_open)
    mock_yaml_dump = mocker.patch("yaml.dump")

    # Входные данные для теста
    dataset_path = "/path/to/dataset"
    class_labels = {0: "ClassA", 1: "ClassB"}
    yaml_path = "/path/to/config.yaml"

    # Вызываем тестируемую функцию
    create_config_file(dataset_path, class_labels, yaml_path)

    # Проверяем, что open был вызван с правильными аргументами
    mock_open.assert_called_once_with(yaml_path, "w", encoding="utf-8")

    # Проверяем, что yaml.dump был вызван дважды
    assert mock_yaml_dump.call_count == 2

    # Проверяем, что первый вызов yaml.dump записал путь к данным
    mock_yaml_dump.assert_any_call(
        {
            "path": dataset_path,
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
        },
        mock_open(),
        Dumper=yaml.SafeDumper,
        default_flow_style=False,
        allow_unicode=True,
    )

    # Проверяем, что второй вызов yaml.dump записал метки классов
    mock_yaml_dump.assert_any_call(
        {"names": class_labels},
        mock_open(),
        Dumper=yaml.SafeDumper,
        default_flow_style=False,
        allow_unicode=True,
    )


def test_create_config_file_file_creation(mocker):
    # Мокируем функцию open
    mock_open = mocker.mock_open()
    mocker.patch("builtins.open", mock_open)

    # Входные данные для теста
    dataset_path = "/path/to/dataset"
    class_labels = {0: "ClassA", 1: "ClassB"}
    yaml_path = "/path/to/new_config.yaml"

    # Вызываем тестируемую функцию
    create_config_file(dataset_path, class_labels, yaml_path)

    # Проверяем, что open был вызван с правильными аргументами
    # (т.е. файл был открыт для записи)
    mock_open.assert_called_once_with(yaml_path, "w", encoding="utf-8")

    # Проверяем, что файл был записан
    handle = mock_open()
    handle.write.assert_called()  # Убедимся, что была запись в файл


def test_create_config_file_io_error(mocker):
    # Мокируем функцию open, чтобы она выбрасывала IOError
    mock_open = mocker.mock_open()
    mock_open.side_effect = IOError("Unable to write file")
    mocker.patch("builtins.open", mock_open)

    # Мокируем print для проверки вывода ошибки
    mock_print = mocker.patch("builtins.print")

    # Входные данные для теста
    dataset_path = "/path/to/dataset"
    class_labels = {0: "ClassA", 1: "ClassB"}
    yaml_path = "/path/to/faulty_config.yaml"

    # Вызываем тестируемую функцию
    create_config_file(dataset_path, class_labels, yaml_path)

    # Проверяем, что open вызвал IOError
    mock_open.assert_called_once_with(yaml_path, "w", encoding="utf-8")

    # Проверяем, что сообщение об ошибке было напечатано
    mock_print.assert_called_once_with("Error creating YAML file: Unable to write file")
