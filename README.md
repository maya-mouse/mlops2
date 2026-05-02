# Лабораторна робота №2: CI/CD та ML API (Варіант 19)

[![CI](https://github.com/maya-mouse/mlops2/actions/workflows/ci.yml/badge.svg)](https://github.com/maya-mouse/mlops2/actions/workflows/ci.yml)

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

2. Встановіть залежності:
   ```bash
   pip install -r requirements.txt

3. Натренуйте модель:
   ```bash
   python -m ml.train

4. Запустіть сервер:
   ```bash
   uvicorn app.maiin:app --reload

Посилання на деплой

- Публічна адреса сервісу: https://mlops2-maya.onrender.com
- Статус моделі: https://mlops2-maya.onrender.com/health
- Документація API (Swagger): https://mlops2-maya.onrender.com/docs