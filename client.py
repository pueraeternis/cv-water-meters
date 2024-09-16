import asyncio

import httpx


# Отправка изображения и получение его обратно
async def send_image_return(file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as image_file:
            files = {"image": (image_file.name, image_file, "image/jpeg")}
            response = await client.post(
                "http://127.0.0.1:80/image/visualize", files=files
            )
            if response.status_code == 200:
                with open("returned_image.jpg", "wb") as f:
                    f.write(response.content)
                print(
                    "Изображение успешно получено обратно и сохранено как 'returned_image.jpg'"
                )
            else:
                print(f"Ошибка: {response.status_code} - {response.text}")


# Отправка изображения и получение описания
async def send_image_describe(file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as image_file:
            files = {"image": (image_file.name, image_file, "image/jpeg")}
            response = await client.post(
                "http://127.0.0.1:80/image/readings", files=files
            )
            if response.status_code == 200:
                print(response.json())
            else:
                print(f"Ошибка: {response.status_code} - {response.text}")


# Запуск клиентов
if __name__ == "__main__":
    # Укажите путь к вашему изображению
    image_path = "data/images/id_12_value_414_676.jpg"

    # Отправить изображение и получить его обратно
    asyncio.run(send_image_return(image_path))

    # Отправить изображение и получить описание
    asyncio.run(send_image_describe(image_path))
