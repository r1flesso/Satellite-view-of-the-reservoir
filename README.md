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

## Использование
### Обучение
Для обучения самой лучшей модели на данных clear_data.csv и сохранения ее в MLflow Вам достаточно запустить файл MY_BEST_MODEL.py:
```bash
python MY_BEST_MODEL.py
```
