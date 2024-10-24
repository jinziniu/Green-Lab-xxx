from sklearnex import patch_sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import psutil  # Import psutil to monitor CPU and memory usage
from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain

# 使用 sklearnex 的加速
patch_sklearn()

# 获取当前进程对象，用于后续获取内存使用情况
process = psutil.Process()

# 从 CSV 文件加载数据，确保文件路径正确
df = pd.read_csv('IndiansDiabetes.csv')  # 请确保路径正确

# 提取特征数据和目标标签（'Outcome' 作为标签）
X = df.drop('Outcome', axis=1)  # 删除 'Outcome' 列，保留其他特征
y = df['Outcome']  # 使用 'Outcome' 作为目标标签

# 填补缺失值，使用均值填补
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# SVM 模型实例化（使用线性核）
model = SVC(kernel='linear')

# 使用 pyJoules 来测量能耗
@measure_energy(domains=[RaplPackageDomain(0), RaplCoreDomain(0)])
def run_training():
    # 训练模型
    model.fit(X_train, y_train)

# 记录 CPU 和内存使用率及训练的开始时间
start_cpu_percent = psutil.cpu_percent(interval=1)  # 1 秒时间间隔，稳定 CPU 使用率计算
start_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录初始内存使用情况，单位为 MB
start_time = time.time()  # 记录开始时间

# 执行模型训练并测量能耗
energy_measurement = run_training()

# 模型预测
y_pred = model.predict(X_test)

# 记录结束时间和资源使用情况
end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=1)  # 设置相同时间间隔
end_memory_usage = process.memory_info().rss / (1024 ** 2)  # 记录结束时的内存使用情况，单位为 MB

# 计算运行时间
execution_time = end_time - start_time

# 计算 CPU 使用率变化
cpu_usage = (start_cpu_percent + end_cpu_percent) / 2  # 取 CPU 使用率平均值

# 计算内存使用变化
memory_usage_diff = end_memory_usage - start_memory_usage

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出运行时间、CPU 使用率、内存使用情况和准确率
print(f"Execution Time (sklearnex): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")

# 输出能耗测量结果
for domain in energy_measurement:
    print(f"Energy consumption for {domain.name}: {domain.energy:.2f} Joules")