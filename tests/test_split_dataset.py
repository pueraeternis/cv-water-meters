import os

import pytest

from src.split_dataset import train_test_split  # Замените mymodule на имя вашего модуля

# Тест функции train_test_split


def test_train_test_split_proportions(mocker, tmpdir):
    # Мокаем os.makedirs и shutil.copy
    mock_makedirs = mocker.patch("os.makedirs")
    mock_copy = mocker.patch("shutil.copy")

    # Задаем тестовые данные
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")
    output_dir = tmpdir.mkdir("output")

    # Создаем фейковые изображения и метки
    for i in range(10):
        image_dir.join(f"image_{i}.jpg").write("")
        label_dir.join(f"image_{i}.txt").write("")

    # Запускаем функцию с пропорциями 0.7, 0.15, 0.15

    train_test_split(
        str(image_dir), str(label_dir), str(output_dir), split_ratios=(0.7, 0.15, 0.15)
    )

    # Проверяем, что директории были созданы
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "train", "images"), exist_ok=True
    )
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "train", "labels"), exist_ok=True
    )
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "val", "images"), exist_ok=True
    )
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "val", "labels"), exist_ok=True
    )
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "test", "images"), exist_ok=True
    )
    mock_makedirs.assert_any_call(
        os.path.join(output_dir, "test", "labels"), exist_ok=True
    )

    # Проверяем, что копирование файлов произошло для всех частей
    assert mock_copy.call_count == 20  # 10 изображений + 10 меток

    # Проверка распределения: 7 для train, 1 для val, 2 для test
    train_files = mock_copy.call_args_list[
        :14
    ]  # Первые 14 вызовов для train (7 изображений и 7 меток)
    val_files = mock_copy.call_args_list[
        14:16
    ]  # 2 вызова для val (1 изображение и 1 метка)
    test_files = mock_copy.call_args_list[
        16:
    ]  # Оставшиеся 4 вызова для test (2 изображения и 2 метки)

    assert len(train_files) == 14
    assert len(val_files) == 2
    assert len(test_files) == 4


def test_train_test_split_value_error(mocker, tmpdir):

    # Задаем тестовые данные
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")
    output_dir = tmpdir.mkdir("output")

    # Создаем фейковые изображения и метки (специально разное количество)
    for i in range(10):
        image_dir.join(f"image_{i}.jpg").write("")
    for i in range(9):  # Создаем на одно меньше меток
        label_dir.join(f"image_{i}.txt").write("")

    # Проверка на ошибку несоответствия количества изображений и меток
    with pytest.raises(ValueError, match="The number of images and labels must match"):
        train_test_split(
            str(image_dir),
            str(label_dir),
            str(output_dir),
            split_ratios=(0.7, 0.15, 0.15),
        )


def test_train_test_split_invalid_ratio(mocker, tmpdir):

    # Задаем тестовые данные
    image_dir = tmpdir.mkdir("images")
    label_dir = tmpdir.mkdir("labels")
    output_dir = tmpdir.mkdir("output")

    # Создаем фейковые изображения и метки
    for i in range(10):
        image_dir.join(f"image_{i}.jpg").write("")
        label_dir.join(f"image_{i}.txt").write("")

    # Проверка на ошибку неверной суммы пропорций
    with pytest.raises(ValueError, match="The sum of the shares must be equal to 1"):
        train_test_split(
            str(image_dir),
            str(label_dir),
            str(output_dir),
            split_ratios=(0.6, 0.2, 0.1),
        )
