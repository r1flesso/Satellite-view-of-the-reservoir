from xgb_gbtree import XGBoost_Model
import xgboost as xgb
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

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)


metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

if __name__ == '__main__':
    for metric in metrics:
        Obj = XGBoost_Model(
            dtrain, dval, dtest, X_test, y_test, 'dart', metric
        )
        Obj.objective_optuna()
