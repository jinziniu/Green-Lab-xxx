import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain

# Get current process object to track memory usage
process = psutil.Process()

# Load data from CSV file, ensure the path is correct
df = pd.read_csv('WineQT.csv')  # Make sure the file path and name are correct

# Drop 'Id' and 'quality' columns, keep only features for clustering
X = df.drop(['Id', 'quality'], axis=1)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN instantiation
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples based on data characteristics

# Record CPU and memory usage as well as the start time
start_cpu_percent = psutil.cpu_percent(interval=None)  # Record initial system-wide CPU usage
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # Record initial memory usage in MB
start_time = time.time()  # Record start time

# 使用 EnergyContext 来测量能耗并运行 DBSCAN
with EnergyContext(domains=[RaplPackageDomain(0)], start_tag='run_dbscan') as ctx:
    # Train DBSCAN model
    dbscan.fit(X_scaled)
    ctx.record(tag='DBSCAN_completed')

# Record end time, CPU usage, and memory usage
end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)  # Record final system-wide CPU usage
end_memory_usage = process.memory_info().rss / (1024 ** 2)  # Record final memory usage in MB

# Calculate execution time
execution_time = end_time - start_time

# Calculate average CPU usage during execution
cpu_usage_avg = (start_cpu_percent + end_cpu_percent) / 2  # Average CPU usage during execution

# Calculate memory usage difference
memory_usage_diff = end_memory_usage - start_memory_usage

# Get clustering labels (-1 indicates noise points)
labels = dbscan.labels_

# Calculate silhouette score if there are at least two clusters
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    silhouette_avg = -1
    print("Silhouette Score could not be calculated because there is only one cluster.")

# Output execution time, CPU usage, and memory usage change
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage_avg:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
