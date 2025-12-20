from random_forest_sqrt import RandomForestClassifier_Model
import itertools
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_DIR = Path(file).resolve().parents[3]
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


# Прописываем списки bootstrap_list и metrics, по которым будем бежать циклом
bootstrap_list = [True, False]
metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

if __name__ == '__main__':
    for bootstrap, metric in itertools.product(bootstrap_list, metrics):
        # Создаем экземпляр класса с max_features=None
        Obj = RandomForestClassifier_Model(
            X_train, X_val, X_test, y_train, y_val, y_test, None, bootstrap, metric
        )
        # Вызываем метод для обучения
        Obj.objective_optuna()
