FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-app.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-app.txt

COPY API/ ./API/
COPY Models/ ./Models/
COPY Data/Cleaned/ ./Data/Cleaned/
COPY ["Data/Feature Engineered/", "./Data/Feature Engineered/"]
COPY ["Data/Training Data/", "./Data/Training Data/"]

EXPOSE 8000 8501