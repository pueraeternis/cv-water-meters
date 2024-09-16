# 
FROM python:3.10

# 
WORKDIR /app

# 
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
RUN poetry add python-multipart

# 
COPY ./app /app

#
EXPOSE 80

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

