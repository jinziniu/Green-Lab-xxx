import pandas as pd
from sklearnex import patch_sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain

# 使用 sklearnex 的加速
patch_sklearn()

# 获取当前进程对象，用于后续获取 CPU 和内存使用情况
process = psutil.Process()

# 从 CSV 文件加载数据，确保路径正确
df = pd.read_csv('IndiansDiabetes.csv')  # 确保路径正确

# 提取特征数据和目标标签
X = df.drop('Outcome', axis=1)  # 删除 'Outcome' 列，保留其他特征
y = df['Outcome']  # 使用 'Outcome' 作为目标标签

# 填补缺失值，使用均值填补
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Gradient Boosting Classifier 实例化
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 记录 CPU 和内存使用率及训练的开始时间
start_time = time.time()  # 记录开始时间
start_cpu_percent = psutil.cpu_percent(interval=None)  # 捕获训练前的 CPU 使用率
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录初始内存使用情况（MB）

# 使用 EnergyContext 来测量能耗并执行模型训练
with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)], start_tag='start') as ctx:
    model.fit(X_train, y_train)
    ctx.record(tag='Training_completed')  # 手动记录训练结束的能耗

# 模型预测
y_pred = model.predict(X_test)

# 记录结束时间、CPU 和内存使用率
end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)  # 捕获训练后的 CPU 使用率
end_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录最终内存使用情况（MB）

# 计算运行时间
execution_time = end_time - start_time

# 计算 CPU 使用率变化
cpu_usage_avg = (start_cpu_percent + end_cpu_percent) / 2  # 取平均 CPU 使用率

# 计算内存使用变化
memory_usage_diff = end_memory_usage - start_memory_usage

# 计算准确率和其他评估指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 输出运行时间、CPU 和内存使用情况
print(f"Execution Time (sklearnex, Gradient Boosting): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage_avg:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

