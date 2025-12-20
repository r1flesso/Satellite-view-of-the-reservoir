import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score



BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / 'data' / 'clear_data.csv'

# Переводим нашу csv таблицу в формат pd.DataFrame
df = pd.read_csv(DATA_PATH)


X = df.drop(columns=['type'])
y_true = df['type']


# Пишем функцию для обучения модели кластеризации
def run_clustering_experiments(metric, linkage):
    # Создаем наш run и логируем теги метрики и т.п.
    with mlflow.start_run(
        run_name=f'Agglomerative_Clustering_with_{linkage}_linkage_n_{metric}_metric'
    ):
        mlflow.autolog()    # MLflow будет за нас логировать

        model = AgglomerativeClustering(
            n_clusters=4,       # Количество кластеров
            metric=metric,      # Метрика расстояния между точками
            linkage=linkage     # Метод объединения кластеров
        )

        # Обучение и предсказание меток кластеров
        labels = model.fit_predict(X)

        # Логирование метаданных эксперимента
        mlflow.set_tags({
            'model_type': 'Hierarchical_Clustering',
            'data_size': f'{X.shape[0]} samples',
            'algorithm_type': 'clustering',
            'task_type': 'unsupervised_learning',
            'n_clusters': '4',
            'feature_scaling': 'none',
            'linkage_method': linkage,
            'distance_metric': metric
        })

        # Вычисление метрики качества кластеризации
        # Adjusted Rand Index сравнивает предсказанные и истинные метки
        ari = adjusted_rand_score(y_true, labels)
        mlflow.log_metrics({'adjusted_rand_score': ari})    # Логируем метрики
        print(ari)


# Прописываем список linkage, который будем
linkage_methods = ['ward', 'complete', 'average', 'single']

# Устанавливаем эксперимент в MLflow
mlflow.set_experiment('Agglomerative_Clustering')

# Бежим по циклу и для каждой комбинации metric и linkage запускаем эксперимент кластеризации
for linkage in linkage_methods:
    if linkage == 'ward':
        run_clustering_experiments('euclidean', 'ward')
    else:
        metrics = ['euclidean', 'manhattan', 'cosine']
        for metric in metrics:
            run_clustering_experiments(metric, linkage)
