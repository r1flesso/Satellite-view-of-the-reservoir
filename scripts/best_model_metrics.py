import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path


MLFLOW_PATH = Path(__file__).parent / 'src' / 'models' / 'classifical' / 'mlruns'
mlflow.set_tracking_uri(f'file://{MLFLOW_PATH.absolute()}')

client = MlflowClient()
run = client.get_run('8c577bd5d33d403f80562358d09dbea7')

print('Метрики:')
for metric, value in run.data.metrics.items():
    print(f'{metric}: {value:.4f}')
