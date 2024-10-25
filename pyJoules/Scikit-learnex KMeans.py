from sklearnex import patch_sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain


patch_sklearn()


process = psutil.Process()


df = pd.read_csv('WineQT.csv')


X = df.drop(['Id', 'quality'], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)


start_cpu_percent = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)
start_time = time.time()


with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)]) as ctx:
    kmeans.fit(X_scaled)
    ctx.record(tag='KMeans Completed')


end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)


execution_time = end_time - start_time


cpu_usage = (start_cpu_percent + end_cpu_percent) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)


print(f"Execution Time (scikit-learnex, K-Means): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Silhouette Score: {silhouette_avg:.4f}")

