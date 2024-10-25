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


patch_sklearn()


process = psutil.Process()


df = pd.read_csv('IndiansDiabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


start_time = time.time()
start_cpu_percent = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)

with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)], start_tag='start') as ctx:
    model.fit(X_train, y_train)
    ctx.record(tag='Training_completed')


y_pred = model.predict(X_test)


end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)


execution_time = end_time - start_time


cpu_usage_avg = (start_cpu_percent + end_cpu_percent) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"Execution Time (sklearnex, Gradient Boosting): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage_avg:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

