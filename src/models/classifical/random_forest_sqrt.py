import mlflow.sklearn
import pandas as pd
from pathlib import Path
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score,\
    precision_score, cohen_kappa_score, matthews_corrcoef
import itertools
import matplotlib
matplotlib.use('Agg')
# Устанавливаем бэкенд 'Agg' для Matplotlib - режим без GUI (графического интерфейса)
# Это позволяет генерировать и сохранять графики в файлы (PNG/PDF) без отображения на экране
# Особенно полезно при работе на серверах, в облачных средах или контейнерах, где нет графической подсистемы




BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / 'data' / 'clear_data.csv'

# Переводим нашу csv таблицу в формат pd.DataFrame
df = pd.read_csv(DATA_PATH, sep=';', decimal=',')


X = df.drop(columns=['type'])
y = df['type']

# Разделяем данные на тренировочную, тестовую и валидационную выборки
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.66, stratify=y_val_test, random_state=42
)


# Класс для обучения и оптимизации RandomForest
class RandomForestClassifier_Model():
    def __init__(
        self, X_train, X_val, X_test, y_train, y_val, y_test, max_features, bootstrap, metric
    ):
        # Инициализация данных и параметров
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.max_features = max_features  # Количество признаков для разделения
        self.bootstrap = bootstrap  # Использовать ли бутстрап выборки
        self.metric = metric  # Метрика для оптимизации

    def objective_optuna(self):
        # Функция вычисления выбранной метрики
        def calculate_metric_score(y_true, y_pred, metric):
            try:
                if metric == 'balanced_accuracy':
                    return balanced_accuracy_score(y_true, y_pred)
                elif metric == 'f1':
                    return f1_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'recall':
                    return recall_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'precision':
                    return precision_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'cohen_kappa':
                    return cohen_kappa_score(y_true, y_pred)
                elif metric == 'matthews_corrcoef':
                    return matthews_corrcoef(y_true, y_pred)
            except:
                return 0.0  # Возвращаем 0 при ошибке


        # Целевая функция для Optuna
        def objective(trial):
            # Гиперпараметры для оптимизации
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 275),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 50, 1500),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 950),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'max_samples': trial.suggest_float('max_samples', 0.7, 1.0) if self.bootstrap else None,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)  # Создание модели

            model.fit(self.X_train, self.y_train)  # Обучение

            # Предсказание и оценка на валидации
            y_val_pred = model.predict(self.X_val)
            score = calculate_metric_score(self.y_val, y_val_pred, self.metric)

            if score is None:
                return 0.0

            return score


        study = optuna.create_study(direction='maximize')  # Создание исследования Optuna
        study.optimize(objective, n_trials=20, show_progress_bar=True)  # Запуск оптимизации

        best_params = study.best_params  # Лучшие параметры


        # Создаем эксперимент MLflow, в который будем логировать теги, метрики, модель т.д.
        mlflow.set_experiment('Random_Forest_Classifier')

        # Создаем запуск MLflow, указываем имя run
        with mlflow.start_run(
            run_name=f'RFC_with_{self.max_features}_n_{self.bootstrap}_bootstrap_n_{self.metric}'
        ) as run:
            # Логируем параметры модели
            mlflow.log_params({
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'optimized_metric': self.metric,
                **best_params
            })

            # Обучение финальной модели с лучшими параметрами
            forest = RandomForestClassifier(**best_params)
            forest.fit(self.X_train, self.y_train)

            # Предсказание на тестовой выборке
            predict_labels = forest.predict(self.X_test)

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

            # Вывод метрик в консоль
            print(f'===== {self.max_features}_n_{self.bootstrap}_n_{self.metric} =====')
            print(f'Balanced accuracy: {bacc:.4f}')
            print(f'F1-score: {f1:.4f}')
            print(f'Recall: {rec:.4f}')
            print(f'Precision: {pre:.4f}')
            print(f'Cohen Kappa: {kappa:.4f}')
            print(f'Matthews Correlation: {mcc:.4f}')

            # Подготовка данных для оценки в MLflow
            test_df = self.X_test.copy()
            test_df['type'] = self.y_test
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
                model = f'runs:/{run.info.run_id}/model',
                data=test_df,
                targets='type',
                model_type='classifier',
                evaluators=['default'],
            )

            # Установка тегов
            mlflow.set_tags({
                'model_type': 'RandomForest',
                'optimization_metric': self.metric,
                'criterion': self.max_features,
                'splitter': self.bootstrap,
                'task_type': 'territory_classification',
                'data_size': 'all_samples'
            })


# Определяем комбинации bootstrap и metric для перебора
bootstrap_list = [True, False]
metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

# Основной блок выполнения
if __name__ == '__main__':
    # Бежим по циклу, перебираем каждые комбинации bootstrap и metric
    for bootstrap, metric in itertools.product(bootstrap_list, metrics):
        # Создаем объект класса с текущими параметрами
        Obj = RandomForestClassifier_Model(
            X_train, X_val, X_test, y_train, y_val, y_test, 'sqrt', bootstrap, metric
        )
        # Запускаем оптимизацию и логирование
        Obj.objective_optuna()
