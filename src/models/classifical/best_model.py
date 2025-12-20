import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score,\
    precision_score, cohen_kappa_score, matthews_corrcoef
from random_forest_sqrt import X_train, y_train, X_test, y_test, X_val, y_val



def calculate_metric_score(y_true, y_pred):
    try:
        return matthews_corrcoef(y_true, y_pred)
    except:
        return 0.0  # Возвращаем 0 при ошибке


weights = {
    0: 10.0,
    1: 0.8,
    2: 0.5,
    3: 2.5
}


def objective(trial):
    # Гиперпараметры для оптимизации
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 190, 375),
        'max_depth': trial.suggest_int('max_depth', 3, 14),
        'min_samples_split': trial.suggest_int('min_samples_split', 50, 600),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 600),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_features': 'log2',
        'bootstrap': False,
        'class_weight': weights,
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)  # Создание модели

    model.fit(X_train, y_train)  # Обучение

    # Предсказание и оценка на валидации
    y_val_pred = model.predict(X_val)
    score = calculate_metric_score(y_val, y_val_pred)

    if score is None:
        return 0.0

    return score


study = optuna.create_study(direction='maximize')  # Создание исследования Optuna
study.optimize(objective, n_trials=40, show_progress_bar=True)  # Запуск оптимизации

best_params = study.best_params  # Лучшие параметры
best_params['class_weight'] = weights

# Создаем эксперимент MLflow, в который будем логировать теги, метрики, модель т.д.
mlflow.set_experiment('MY_BEST_MODEL')

# Создаем запуск MLflow, указываем имя run
with mlflow.start_run(
    run_name=f'With_custom_class_weight'
) as run:
    # Логируем параметры модели
    mlflow.log_params({
        **best_params
    })

    # Обучение финальной модели с лучшими параметрами
    forest = RandomForestClassifier(**best_params)
    forest.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    predict_labels = forest.predict(X_test)

    # Вычисление всех метрик
    bacc = balanced_accuracy_score(y_test, predict_labels)
    f1 = f1_score(y_test, predict_labels, average='weighted')
    rec = recall_score(y_test, predict_labels, average='weighted')
    pre = precision_score(y_test, predict_labels, average='weighted')
    kappa = cohen_kappa_score(y_test, predict_labels)
    mcc = matthews_corrcoef(y_test, predict_labels)

    # Логирование метрик в MLflow
    mlflow.log_metrics({
        'balanced_accuracy': bacc,
        'f1_score': f1,
        'recall': rec,
        'precision': pre,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc
    })

    # Подготовка данных для оценки в MLflow
    test_df = X_test.copy()
    test_df['type'] = y_test
    test_df = test_df.astype(float)

    input_example = X_train.iloc[0:1]  # Пример данных для модели

    # Регистрация модели
    mlflow.sklearn.log_model(forest, name='model', input_example=input_example)

    # evaluate():
    # 1. Запускает автоматическую оценку сохраненной модели на тестовых данных
    # 2. Вычисляет стандартные метрики классификации (accuracy, precision, recall и т.д.)
    # 3. Генерирует визуализации: confusion matrix, ROC-кривые
    # 4. Создает готовый отчет о качестве модели
    # 5. Все результаты автоматически сохраняются в MLflow для сравнения экспериментов
    # ...
    mlflow.evaluate(
        model=f'runs:/{run.info.run_id}/model',
        data=test_df,
        targets='type',
        model_type='classifier',
        evaluators=['default'],
    )

    # Установка тегов
    mlflow.set_tags({
        'model_type': 'RandomForest',
        'optimization_metric': 'matthews_corrcoef',
        'criterion': 'log2',
        'splitter': False,
        'task_type': 'territory_classification',
        'data_size': 'all_samples'
    })
