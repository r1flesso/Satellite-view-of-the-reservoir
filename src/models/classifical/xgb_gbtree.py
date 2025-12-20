import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score, precision_score, \
    cohen_kappa_score, matthews_corrcoef,  confusion_matrix,\
     roc_auc_score, precision_recall_curve, auc
)




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

# Переводим данные в DMatrix формат, так как XGBoosting работает с ним
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)


# Класс для модели XGBoost
class XGBoost_Model():
    def __init__(
        self, dtrain, dval, dtest, X_test, y_test, booster, metric
    ):
        # Инициализация данных и параметров модели
        self.dtrain = dtrain
        self.dval = dval
        self.dtest = dtest
        self.X_test = X_test
        self.y_test = y_test
        self.booster = booster
        self.metric = metric


    # Функция для построения матрицы ошибок
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))

        # Отрисовка
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Подписи
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xlabel='Predicted label',
            ylabel='True label',
            title='Confusion Matrix'
        )

        # Значения внутри квадратиков
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        return fig


    # Функция, которая создает DataFrame
    # Колонки этого DataFrame - значения precision, recall, F1 etc.
    # Строки - классы (0, 1, 2, 3)
    def per_class_metrics(self, y_true, y_pred, y_proba):
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        rows = []

        for i, cls in enumerate(labels):
            # для класса cls
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            # example_count
            example_count = tp + fn

            # accuracy по данному классу
            acc = (tp + tn) / cm.sum()

            # recall / precision / f1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )

            # roc_auc
            try:
                roc_auc = roc_auc_score((y_true == cls).astype(int), y_proba[:, i])
            except:
                roc_auc = 0

            # PR-AUC
            try:
                p, r, _ = precision_recall_curve((y_true == cls).astype(int), y_proba[:, i])
                pr_auc = auc(r, p)
            except:
                pr_auc = 0

            rows.append([
                cls, tn, fp, fn, tp, example_count,
                acc, recall, precision, f1, roc_auc, pr_auc
            ])

        df = pd.DataFrame(rows, columns=[
            'positive_class',
            'true_negatives',
            'false_positives',
            'false_negatives',
            'true_positives',
            'example_count',
            'accuracy_score',
            'recall_score',
            'precision_score',
            'f1_score',
            'roc_auc',
            'precision_recall_auc'
        ])

        return df


    def objective_optuna(self):
        # Функция для расчета метрик
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


        # Основная функция оптимизации
        def objective(trial):
            # Словарь параметров (в скобках диапазоны значений)
            params = {
                'booster': self.booster,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 4,
                'max_depth': trial.suggest_int('max_depth', 3, 11),
                'gamma': trial.suggest_float('gamma', 0.0, 3.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'random_state': 42,
                'device': 'cuda'
            }

            # Параметры процесса обучения (training_params):
            # num_boost_round - количество деревьев в ансамбле
            # early_stopping_rounds - остановка если метрика не улучшается N раундов
            training_params = {
                'num_boost_round': trial.suggest_int('num_boost_round', 150, 275),
                'early_stopping_rounds': 18
            }

            # Обучаем модель
            model = xgb.train(
                params,
                dtrain=self.dtrain,
                num_boost_round=training_params['num_boost_round'],
                evals=[(self.dtrain, 'train'), (self.dval, 'val')],
                early_stopping_rounds=training_params['early_stopping_rounds'],
                evals_result={},
                verbose_eval=False
            )

            # Прогнозирование и оценка метрики
            y_pred_proba = model.predict(self.dtest)
            score = calculate_metric_score(self.y_test, y_pred_proba, self.metric)

            return score


        # Оптимизация гиперпараметров с помощью Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=75, show_progress_bar=True)

        # Лучшие параметры после оптимизации
        best_params = study.best_params

        best_training_params = {
            'num_boost_round': best_params.pop('num_boost_round'),
            'early_stopping_rounds': 18
        }


        # Логирование результатов в MLflow
        mlflow.set_experiment('XGBoosting_Classifier')

        # Создаем запуск MLflow, указываем имя run
        with mlflow.start_run(
            run_name=f'XGBoost_with_{self.booster}_booster_n_{self.metric}_metric'
        ):
            mlflow.log_params({
                'booster': self.booster,
                'optimized_metric': self.metric,
                **best_params,
                **best_training_params
            })

            # Обучаем модель с лучшими параметрами
            best_model = xgb.train(
                {
                    **best_params,
                    'booster': self.booster,
                    'objective': 'multi:softprob',
                    'num_class': 4
                },
                self.dtrain,
                num_boost_round=best_training_params['num_boost_round'],
                early_stopping_rounds=best_training_params['early_stopping_rounds'],
                evals=[(self.dtrain, 'train'), (self.dval, 'val')],
                verbose_eval=False
            )

            # Прогнозируем с лучшей моделью
            y_pred_proba = best_model.predict(self.dtest)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Рассчитываем метрики для лучшей модели
            bacc = balanced_accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            pre = precision_score(self.y_test, y_pred, average='weighted')
            kappa = cohen_kappa_score(self.y_test, y_pred)
            mcc = matthews_corrcoef(self.y_test, y_pred)

            # Логируем метрики
            mlflow.log_metrics({
                'balanced_accuracy': bacc,
                'f1_score': f1,
                'recall': rec,
                'precision': pre,
                'cohen_kappa': kappa,
                'matthews_corrcoef': mcc
            })

            # Выводим метрики в консоль
            print(f'===== {self.booster}_n_{self.metric} =====')
            print(f'Balanced accuracy: {bacc:.4f}')
            print(f'F1-score: {f1:.4f}')
            print(f'Recall: {rec:.4f}')
            print(f'Precision: {pre:.4f}')
            print(f'Cohen Kappa: {kappa:.4f}')
            print(f'Matthews Correlation: {mcc:.4f}')

            # Регистрируем модель
            mlflow.xgboost.log_model(xgb_model=best_model, name='model')

            # Логируем теги
            mlflow.set_tags({
                'model_type': 'XGBoosting',
                'optimization_metric': self.metric,
                'booster': self.booster,
                'task_type': 'territory_classification',
                'data_size': 'all_samples'
            })

            # Строим матрицу ошибок
            fig = self.plot_confusion_matrix(self.y_test, y_pred)
            mlflow.log_figure(fig, f'confusion_matrix_{self.booster}_{self.metric}.png')
            plt.close(fig)

            # Расчет метрик
            df = self.per_class_metrics(self.y_test, y_pred, y_pred_proba)

            # Сохраняем CSV (перезаписывается)
            csv_path = 'per_class_metrics.csv'
            df.to_csv(csv_path, index=False)

            # Логируем в MLflow
            mlflow.log_artifact(csv_path)


# Метрики для оптимизации
metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

if __name__ == '__main__':
    # Для каждой метрики проводим оптимизацию
    for metric in metrics:
        Obj = XGBoost_Model(
            dtrain, dval, dtest, X_test, y_test, 'gbtree', metric
        )
        Obj.objective_optuna()
