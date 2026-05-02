# Лабораторна робота №2: CI/CD та ML API (Варіант 19)

![CI Status](https://github.com/maya-mouse/ml-api-lab2/actions/workflows/ci.yml/badge.svg)

## Опис проєкту
Цей проєкт реалізує наскрізний MLOps-конвеєр для класифікації ірисів (Iris dataset). Сервіс побудований на FastAPI, контейнеризований за допомогою Docker та розгорнутий на платформі Render.

## Стек технологій
* **Мова**: Python 3.11
* **ML**: Scikit-learn, Joblib
* **API**: FastAPI, Pydantic, Uvicorn
* **CI/CD**: GitHub Actions
* **Контейнеризація**: Docker
* **Деплой**: Render

## Як запустити локально

1. Створіть віртуальне середовище:
   ```bash
   python -m venv venv
   source venv/bin/activate  # для Linux/macOS
   # або
   venv\Scripts\activate     # для Windows