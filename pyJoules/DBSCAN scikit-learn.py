import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time
import psutil
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain


process = psutil.Process()


df = pd.read_csv('WineQT.csv')


X = df.drop(['Id', 'quality'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN instantiation
dbscan = DBSCAN(eps=0.5, min_samples=5)


start_cpu_percent = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)
start_time = time.time()  #


with EnergyContext(domains=[RaplPackageDomain(0)], start_tag='run_dbscan') as ctx:

    dbscan.fit(X_scaled)
    ctx.record(tag='DBSCAN_completed')


end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)


execution_time = end_time - start_time


cpu_usage_avg = (start_cpu_percent + end_cpu_percent) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


labels = dbscan.labels_


if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    silhouette_avg = -1
    print("Silhouette Score could not be calculated because there is only one cluster.")


print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage_avg:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
