import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import time
import psutil  # Import psutil to monitor CPU and memory usage
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain

# 获取当前进程对象，用于后续获取内存使用情况
process = psutil.Process()

# 从 CSV 文件加载数据，确保路径正确
df = pd.read_csv('HousingData.csv')  # 请确保路径正确

# 提取特征数据和目标标签（'MEDV' 作为目标变量）
X = df.drop('MEDV', axis=1)  # 删除 'MEDV' 列，保留其他特征
y = df['MEDV']  # 使用 'MEDV' 作为目标标签

# 填补缺失值，使用均值填补
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Linear Regression 实例化
model = LinearRegression()

# 记录 CPU 和内存使用率及训练的开始时间
start_cpu_percent = psutil.cpu_percent(interval=None)  # 无时间间隔
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录初始内存使用情况，单位为 MB
start_time = time.time()  # 记录开始时间

# 使用 EnergyContext 来测量能耗
with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)]) as ctx:
    model.fit(X_train, y_train)  # 训练模型
    ctx.record(tag='Training Completed')  # 手动记录能耗数据

# 模型预测
y_pred = model.predict(X_test)

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

# 计算均方误差 (MSE) 和 R² 分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出运行时间、CPU 使用率、内存使用情况、MSE 和 R² 分数
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

