import mlflow.sklearn
import optuna
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score,\
    precision_score, cohen_kappa_score, matthews_corrcoef
# cohen_kappa_score - Учитывает случайное угадывание, особенно важно при несбалансированных классах
# (разные типы территории могут быть представлены неравномерно)
# matthews_corrcoef - Работает хорошо даже при сильном дисбалансе классов, учитывает все ячейки матрицы ошибок
# (идеально для многоклассовой классификации)




BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / 'data' / 'clear_data.csv'

# Переводим нашу csv таблицу в формат pd.DataFrame
df = pd.read_csv(DATA_PATH)


X = df.drop(columns=['type'])
y = df['type']

# Разделяем данные на тренировочную, тестовую и валидационную выборки
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.66, stratify=y_val_test, random_state=42
)

# Уменьшаем выборку, так как на больших данных SVC очень медленный
X_train_small = X_train[:20000]
y_train_small = y_train[:20000]


# Создаем функцию внутри функции
# В create_objective будем передавать метрику, которую будем улучшать
# objective - функция, которая подбирает лучшие гиперпараметры для нашей модели
def create_objective(metric):
    def objective(trial):
        # Создаем словарь параметров для нашей модели
        # В круглых скобках прописываем либо список, либо границы значений
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        # Создаем нашу модель и передаем словарь с параметрами
        svc = SVC(**params, random_state=42)
        svc.fit(X_train_small, y_train_small)  # Обучаем на тренировочных данных
        y_pred = svc.predict(X_val)  # Предсказание на валидации

        # Вычисляем метрику качества на валидационной выборке
        # В зависимости от выбранной метрики используем соответствующую функцию
        if metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_val, y_pred)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, average='weighted')
        elif metric == 'f1':
            score = f1_score(y_val, y_pred, average='weighted')
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, average='weighted')

        return score

    return objective


# Создаем список с метриками, которые поэтапно будем улучшать
metrics = ['balanced_accuracy', 'precision', 'recall', 'f1']

mlflow.set_experiment('SVC')

for metric in metrics:
    with mlflow.start_run(run_name=f'SVC_with_{metric}_metric'):
        objective_func = create_objective(metric)

        # Создаем и запускаем обучение нашей модели
        # Используем pruner=optuna.pruners.MedianPruner() для преждевременной остановки обучения,
        # если нет заметных улучшений модели
        # n_trials=10 - количество попыток подбора гиперпараметров
        # show_progress_bar=True - показывает прогресс-бар для отслеживания процесса оптимизации
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective_func, n_trials=10, show_progress_bar=True)

        # Получаем лучшие параметры из исследования Optuna
        params = study.best_params

        # Добавляем фиксированные параметры для SVC
        params['random_state'] = 42
        params['probability'] = False

        # Обучаем финальную модель на тренировочных данных
        svc = SVC(**params)
        svc.fit(X_train_small, y_train_small)

        # Предсказываем на тестовой выборке
        predict_labels = svc.predict(X_test)

        # Считаем метрики на тестовых данных
        bacc = balanced_accuracy_score(y_test, predict_labels)
        f1 = f1_score(y_test, predict_labels, average='weighted')
        rec = recall_score(y_test, predict_labels, average='weighted')
        pre = precision_score(y_test, predict_labels, average='weighted')
        kappa = cohen_kappa_score(y_test, predict_labels)
        mcc = matthews_corrcoef(y_test, predict_labels)

        # Логируем в mlflow самые лучшие гиперпараметры
        mlflow.log_params(params)

        # Логируем метрики
        mlflow.log_metrics({
            'balanced_accuracy': bacc,
            'f1': f1,
            'recall': rec,
            'precision': pre,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc
        })

        # Регистрируем модель
        mlflow.sklearn.log_model(svc, 'model')

        # Добавляем теги
        mlflow.set_tags({
            'model_type': 'SVC',
            'optimization_metric': metric,
            'data_size': '20k_samples',
            'kernel_type': 'linear',
            'task_type': 'multiclass_classification',
            'feature_scaling': 'none'
        })

        print(f'Balanced accuracy: {bacc:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Recall: {rec:.4f}')
        print(f'Precision: {pre:.4f}')
        print(f'Cohen Kappa: {kappa:.4f}')
        print(f'Matthews Correlation: {mcc:.4f}')


# Пробовал обучить модели на выборке из 32K, 45K, 50K samples,
# но это было невыносимо долго. Модель с 32K samples обучалась на 30-40% и зависала
