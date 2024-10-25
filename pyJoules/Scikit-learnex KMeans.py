from sklearnex import patch_sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain

# 使用 sklearnex 的加速
patch_sklearn()

# 获取当前进程对象，用于后续获取内存使用情况
process = psutil.Process()

# 从 CSV 文件加载数据
df = pd.read_csv('WineQT.csv')  # 请确保路径正确

# 删除 'Id' 和 'quality' 列，只保留用于聚类的特征
X = df.drop(['Id', 'quality'], axis=1)

# 对特征数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 实例化（使用 scikit-learnex 加速）
kmeans = KMeans(n_clusters=3, random_state=42)

# 记录 CPU 和内存使用率及训练的开始时间
start_cpu_percent = psutil.cpu_percent(interval=None)  # 无时间间隔
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录初始内存使用情况，单位为 MB
start_time = time.time()  # 记录开始时间

# 使用 EnergyContext 来测量能耗
with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)]) as ctx:
    kmeans.fit(X_scaled)
    ctx.record(tag='KMeans Completed')  # 手动记录能耗数据

# 记录结束时间和资源使用情况
end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)  # 无时间间隔
end_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录结束时的内存使用情况，单位为 MB

# 计算运行时间
execution_time = end_time - start_time

# 计算 CPU 使用率变化
cpu_usage = (start_cpu_percent + end_cpu_percent) / 2  # 取 CPU 使用率平均值

# 计算内存使用变化
memory_usage_diff = end_memory_usage - start_memory_usage

# 计算轮廓系数
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)

# 输出运行时间、CPU 使用率、内存使用情况和轮廓系数
print(f"Execution Time (scikit-learnex, K-Means): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Silhouette Score: {silhouette_avg:.4f}")

