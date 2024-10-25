import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import time
import psutil  # Import psutil to monitor CPU and memory usage
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain


process = psutil.Process()


df = pd.read_csv('HousingData.csv')


X = df.drop('MEDV', axis=1)
y = df['MEDV']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

model = LinearRegression()


start_cpu_percent = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)
start_time = time.time()


with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)]) as ctx:
    model.fit(X_train, y_train)
    ctx.record(tag='Training Completed')


y_pred = model.predict(X_test)


end_time = time.time()
end_cpu_percent = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)

execution_time = end_time - start_time


cpu_usage = (start_cpu_percent + end_cpu_percent) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

