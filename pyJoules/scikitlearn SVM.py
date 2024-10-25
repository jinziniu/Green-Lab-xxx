import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import time
import psutil  # Import psutil to monitor CPU and memory usage
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain


process = psutil.Process()


df = pd.read_csv('IndiansDiabetes.csv')


X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 填补缺失值，使用均值填补
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


model = SVC(kernel='linear')


start_cpu_percent = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)
start_time = time.time()


with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)]) as ctx:
    model.fit(X_train, y_train)
    ctx.record(tag='SVM Training Completed')


end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)


execution_time = end_time - start_time


cpu_usage = (start_cpu_percent + end_cpu_percent) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")

