import pandas as pd
from sklearnex import patch_sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain

# 使用 sklearnex 的加速
patch_sklearn()

# 获取当前进程对象，用于后续获取内存使用情况
process = psutil.Process()

# 从 CSV 文件加载数据，确保路径正确
df = pd.read_csv('WineQT.csv')  # 确保路径和文件名正确

# 删除 'Id' 和 'quality' 列，只保留用于聚类的特征
X = df.drop(['Id', 'quality'], axis=1)

# 对特征数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN 实例化
dbscan = DBSCAN(eps=0.5, min_samples=5)  # 你可以根据数据特性调整 eps 和 min_samples 参数

# 记录 CPU 和内存使用率及训练的开始时间
start_cpu_percent = psutil.cpu_percent(interval=None)  # 使用 1 秒的时间间隔稳定 CPU 使用率的计算
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录初始内存使用情况（单位：MB）
start_time = time.time()  # 记录开始时间

# 使用 pyJoules 来测量能耗，通过 EnergyContext 手动控制测量
with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)], start_tag='start') as ctx:
    dbscan.fit(X_scaled)
    ctx.record(tag='DBSCAN_completed')  # 手动记录能耗数据

# 记录结束时间、CPU 使用率和内存使用率
end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)  # 使用相同时间间隔稳定 CPU 使用率的计算
end_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录最终内存使用情况（单位：MB）

# 计算运行时间
execution_time = end_time - start_time

# 计算 CPU 使用率变化
cpu_usage = (start_cpu_percent + end_cpu_percent) / 2  # 取 CPU 使用率的平均值

# 计算内存使用变化
memory_usage_diff = end_memory_usage - start_memory_usage

# 获取聚类标签 (-1 表示噪声点)
labels = dbscan.labels_

# 计算轮廓系数，前提是必须有至少两个不同的聚类
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    silhouette_avg = -1
    print("Silhouette Score could not be calculated because there is only one cluster.")

# 输出运行时间、CPU 使用率和内存使用情况
print(f"Execution Time (scikit-learnex, DBSCAN): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")


