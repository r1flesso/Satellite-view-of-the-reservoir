import mlflow.sklearn
import pandas as pd
from pathlib import Path
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score,\
    precision_score, cohen_kappa_score, matthews_corrcoef
import itertools




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


# Класс для модели DecisionTreeClassifier
class DecisionTreeClassifier_Model():
    def __init__(
        self, X_train, X_val, X_test, y_train, y_val, y_test, criterion, splitter, metric
    ):
        # Инициализация данных и параметров модели
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.criterion = criterion
        self.splitter = splitter
        self.metric = metric

    def objective_optuna(self):
        # Функция, которая вычисляет метрики
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
                return 0.0

        # Целевая функция для Optuna
        def objective(trial):
            # Пишем словарь гиперпараметров, в скобках указывая границы значений каждого параметра
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_samples_split': trial.suggest_int('min_samples_split', 50, 2500),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 1500),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 105),
                'criterion': self.criterion,
                'splitter': self.splitter
            }

            # Создаем и обучаем модель
            model = DecisionTreeClassifier(**params, class_weight='balanced')

            model.fit(self.X_train, self.y_train)

            # Делаем предсказания на валидационной выборке
            y_val_pred = model.predict(self.X_val)
            # Вычисляем метрику
            score = calculate_metric_score(self.y_val, y_val_pred, self.metric)

            if score is None:
                return 0.0

            return score

        # Создаем исследование Optuna
        study = optuna.create_study(direction='maximize')
        # Запускаем оптимизацию
        study.optimize(objective, n_trials=70, show_progress_bar=True)

        # Получаем лучшие параметры
        best_params = study.best_params


        # Создаем эксперимент MLflow, в который будем логировать теги, метрики, модель т.д.
        mlflow.set_experiment('Decision_Tree_Classifier')

        # Создаем запуск MLflow, указываем имя run
        with mlflow.start_run(
            run_name=f'DTC_with_{self.splitter}_n_{self.criterion}_n_{self.metric}'
        ) as run:
            # Логируем параметры модели
            mlflow.log_params({
                'criterion': self.criterion,
                'splitter': self.splitter,
                'optimized_metric': self.metric,
                **best_params
            })

            # Создаем и обучаем модель с лучшими гиперпараметрами
            tree = DecisionTreeClassifier(**best_params, class_weight='balanced')
            tree.fit(self.X_train, self.y_train)

            # Делаем предсказания на тестовой выборке
            predict_labels = tree.predict(self.X_test)

            # Вычисляем все метрики
            bacc = balanced_accuracy_score(y_test, predict_labels)
            f1 = f1_score(y_test, predict_labels, average='weighted')
            rec = recall_score(y_test, predict_labels, average='weighted')
            pre = precision_score(y_test, predict_labels, average='weighted')
            kappa = cohen_kappa_score(y_test, predict_labels)
            mcc = matthews_corrcoef(y_test, predict_labels)

            # Логируем метрики в MLflow
            mlflow.log_metrics({
                'balanced_accuracy': bacc,
                'f1_score': f1,
                'recall': rec,
                'precision': pre,
                'cohen_kappa': kappa,
                'matthews_corrcoef': mcc
            })

            # Выводим метрики в консоль для каждой комбинации splitter, criterion и metric
            print(f'===== {self.splitter}_n_{self.criterion}_n_{self.metric} =====')
            print(f'Balanced accuracy: {bacc:.4f}')
            print(f'F1-score: {f1:.4f}')
            print(f'Recall: {rec:.4f}')
            print(f'Precision: {pre:.4f}')
            print(f'Cohen Kappa: {kappa:.4f}')
            print(f'Matthews Correlation: {mcc:.4f}')

            # Создаем DataFrame для оценки модели в MLflow
            test_df = self.X_test.copy()
            test_df['type'] = self.y_test
            test_df = test_df.astype(float)

            # Пример входных данных для модели
            input_example = X_train.iloc[0:1]

            # Регистрируем модель в MLflow
            mlflow.sklearn.log_model(tree, name='model', input_example=input_example)

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

            # Устанавливаем теги для запуска
            mlflow.set_tags({
                'model_type': 'DecisionTree',
                'optimization_metric': self.metric,
                'criterion': self.criterion,
                'splitter': self.splitter,
                'task_type': 'territory_classification',
                'data_size': 'all_samples'
            })


# Определяем комбинации splitter и metric для перебора
splitters = ['best', 'random']
metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

# Основной блок выполнения
if __name__ == '__main__':
    # Бежим по циклу, перебираем каждые комбинации splitter и metric
    for splitter, metric in itertools.product(splitters, metrics):
        # Создаем объект класса с текущими параметрами
        Obj = DecisionTreeClassifier_Model(
            X_train, X_val, X_test, y_train, y_val, y_test, 'gini', splitter, metric
        )
        # Запускаем оптимизацию и логирование
        Obj.objective_optuna()
