import pandas as pd
from sklearnex import patch_sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time
import psutil  # Import psutil for CPU and memory usage tracking
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain


patch_sklearn()


process = psutil.Process()


df = pd.read_csv('HousingData.csv')

X = df.drop('MEDV', axis=1)
y = df['MEDV']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


start_time = time.time()
cpu_usage_start = psutil.cpu_percent(interval=None)
start_memory_usage = process.memory_info().rss / (1024 ** 2)

with EnergyContext(domains=[RaplPackageDomain(0), RaplCoreDomain(0)], start_tag="train_model") as ctx:

    model.fit(X_train, y_train)
    ctx.record(tag="train_completed")


y_pred = model.predict(X_test)


end_time = time.time()
cpu_usage_end = psutil.cpu_percent(interval=None)
end_memory_usage = process.memory_info().rss / (1024 ** 2)


execution_time = end_time - start_time


cpu_usage_avg = (cpu_usage_start + cpu_usage_end) / 2


memory_usage_diff = end_memory_usage - start_memory_usage


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Execution Time (sklearnex, Gradient Boosting): {execution_time:.4f} seconds")
print(f"Average CPU Usage during Execution: {cpu_usage_avg:.2f}%")
print(f"Memory Usage Change: {memory_usage_diff:.2f} MB")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")
