# Satellite-view-of-the-reservoir
Проект, в котором реализовано машинное обучение для классификации типов территорий на табличных данных реального космоснимка (водохранилищ)

## Установка
1. Клонируйте мой репозиторий:
    ```bash
    git clone https://github.com/r1flesso/Satellite-view-of-the-reservoir
    ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Установите необходимые библиотеки:
   ```bash
   pip install -r requirements.txt
   ```
   или для PyTorch:
   ```bash
   pip install -r requirements_pytorch.txt
   ```

## Использование
- Для сбора метрик каждого run запустите файл collect_metrics.py:
```bash
python collect_metrics.py
```
- Для обучения самой лучшей модели на данных clear_data.csv и сохранения ее в MLflow Вам достаточно запустить файл best_model.py:
```bash
python best_model.py
```
- Для вывода метрик самой лучшей модели запустите файл best_model_metrics.py:
```bash
python best_model_metrics.py
```
- Для создание своего docker image используйте файл dockerfile.example в качестве примера

## Структура проекта
- Папка data содержит в себе очищенный датасет, на котором обучаются модели
- В папке notebooks ...
- В папке scripts представлены два файла: best_model_metrics.py и collect_metrics.py
- В папке src/models/classifical представлены все модели машинного обучения, в том числе best_model.py, которая реализована на Random Forest
- Два файла requirements.txt и requirements_pytorch.txt служат для управления зависимостями проекта, каждый из которых включает необходимые библиотеки для различных компонентов. requirements.py содержит стандартные зависимости для работы с моделью, а requirements_pytorch.txt — для установки зависимостей, необходимых для работы с библиотеками PyTorch

## Данные
Датасет предоставлен научным руководителем для учебного проекта. Содержит табличные данные, полученные из космоснимков водохранилищ

## Лицензия
Этот проект распространяется под лицензией MIT
