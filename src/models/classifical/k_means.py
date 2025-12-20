import mlflow.sklearn
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score



BASE_DIR = Path(file).resolve().parents[3]
DATA_PATH = BASE_DIR / 'data' / 'clear_data.csv'

# Переводим нашу csv таблицу в формат pd.DataFrame
df = pd.read_csv(DATA_PATH, sep=';', decimal=',')

y_true = df['type']  # Истинные метки кластеров для оценки

X = df.drop(columns=['type'])

algorithms = ['lloyd', 'elkan']  # Два алгоритма K-Means для сравнения

mlflow.set_experiment('KMeans')  # Создаем эксперимент для кластеризации

for algorithm in algorithms:  # Перебираем оба алгоритма
    # Создаем запуск MLflow, указываем имя run
    with mlflow.start_run(run_name=f'KMeans_with_{algorithm}_algorithm'):
        mlflow.autolog()  # Автоматическое логирование параметров и метрик

        # Создаем модель K-Means с фиксированными параметрами
        kmeans = KMeans(
            n_clusters=4,  # Количество кластеров
            init='k-means++',  # Умная инициализация центроидов
            algorithm=algorithm,  # Выбранный алгоритм (lloyd/elkan)
            random_state=42  # Сажаем зерно для воспроизводимости
        )

        labels = kmeans.fit_predict(X)  # Кластеризация данных

        # Устанавливаем теги для запуска
        mlflow.set_tags({
            'model_type': 'KMeans',
            'data_size': f'{X.shape[0]} samples',
            'algorithm_type': 'clustering',
            'task_type': 'unsupervised_learning',
            'n_clusters': '4',
            'feature_scaling': 'standard_scaler'
        })

        # Оценка качества кластеризации (сравнение с истинными метками)
        ari = adjusted_rand_score(y_true, labels)
        mlflow.log_metrics({'adjusted_rand_score': ari})
        print(ari)  # Вывод метрики в консоль
