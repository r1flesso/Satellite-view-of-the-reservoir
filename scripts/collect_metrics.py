import mlflow
import pandas as pd
from pathlib import Path


# Указываем путь к mlruns
MLFLOW_PATH = Path(__file__).parent / 'src' / 'models' / 'classifical' / 'mlruns'
mlflow.set_tracking_uri(f'file://{MLFLOW_PATH.absolute()}')


all_results = []
exp_names = ['Random_Forest_Classifier', 'Decision_Tree_Classifier',
             'Logistic_Regression', 'SVC', 'XGBoosting_Classifier']

for exp_name in exp_names:
    exp = mlflow.get_experiment_by_name(exp_name)
    
    # Пропускаем если эксперимент не найден
    if exp is None:
        continue

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    for _, run in runs.iterrows():
        result = {
            'experiment': exp_name,
            'run_id': run['run_id'],
            'run_name': run.get('tags.mlflow.runName', ''),
        }

        metric_cols = [col for col in run.index if col.startswith('metrics.')]

        for col in metric_cols:
            metric_name = col.replace('metrics.', '')
            result[metric_name] = run[col]

        all_results.append(result)

df = pd.DataFrame(all_results)

# Проверяем есть ли данные
if not df.empty:
    df.to_csv('mlflow_results.csv', index=False)
    print(f'Сохранено {len(df)} записей в mlflow_results.csv')
else:
    print('Не найдено данных для сохранения')
