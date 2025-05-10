# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# === Step 1: Load and clean dataset ===
df = pd.read_csv("/Users/jay/Desktop/Project-goalcode/Data-sets/Goal-3/MTB-project-goal3.1-new.csv")
df.columns = df.columns.str.strip()

carrier_cols = [
    'passenger_academy', 'passenger_greyhound', 'passenger_coach_usa',
    'passenger_transbridge', 'passenger_peterpan', 'passenger_cj'
]
weather_cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2))
df['Month'] = df['Date'].dt.month
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# === Clean: Replace 0 and NaN in weather columns ===
for col in weather_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df.groupby('Month')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

# === Clean: Replace 0 and NaN in carrier columns ===
for col in carrier_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df.groupby('Month')[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

# === Step 2: Prepare future weather data (2025–2030) ===
np.random.seed(42)
future_dates = pd.date_range("2025-01-01", "2030-12-01", freq='MS')
monthly_weather_avg = df.groupby('Month')[weather_cols].mean().reset_index()

future_weather = pd.DataFrame({
    'Date': future_dates,
    'Month': future_dates.month,
    'Year': future_dates.year
})
future_weather = future_weather.merge(monthly_weather_avg, on='Month', how='left')

# Add small variation to future weather
for col in weather_cols:
    std_dev = df[col].std() * 0.05
    noise = np.random.normal(0, std_dev, size=len(future_weather))
    future_weather[col] += noise
    future_weather[col] = future_weather[col].clip(lower=0)

# === Step 3: Train/test split ===
df_eval = df[df['Year'] == 2024]
df_train = df[df['Year'] < 2024]

forecast_list = []
metrics = []
actual_vs_predicted_rows = []

# === Step 4: Forecasting by carrier ===
for col in carrier_cols:
    print(f"Forecasting {col}...")

    X_train = df_train[weather_cols + ['Year', 'Month']]
    y_train = df_train[col]
    X_eval = df_eval[weather_cols + ['Year', 'Month']]
    y_eval = df_eval[col]
    X_future = future_weather[weather_cols + ['Year', 'Month']]

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    preds_eval = best_model.predict(X_eval)
    preds_future = best_model.predict(X_future)

    # Add sinusoidal variation + noise
    months_since_2025 = np.arange(len(preds_future))
    sin_wave = 1 + 0.05 * np.sin(2 * np.pi * months_since_2025 / 12)
    noise = np.random.normal(1.0, 0.05, size=len(preds_future))
    preds_future *= sin_wave * noise

    # Clip values within historical 5–95 percentile
    min_val = df[df[col] > 0][col].quantile(0.05)
    max_val = df[col].quantile(0.95)
    preds_future = np.clip(preds_future, min_val, max_val)
    preds_future *= 1.1  # Scale forecasts by 10%

    temp_df = future_weather.copy()
    temp_df[col] = preds_future
    forecast_list.append(temp_df[['Date', 'Year', 'Month', col]])

    actual_vs_predicted_rows += [
        {'Date': d, 'Carrier': col, 'Actual': a, 'Predicted': p}
        for d, a, p in zip(df_eval.index, y_eval, preds_eval)
    ]

    mae = mean_absolute_error(y_eval, preds_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, preds_eval))
    mape = np.mean(np.abs((y_eval - preds_eval) / y_eval)) * 100
    metrics.append({'Carrier': col, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape})

    os.makedirs("downloads/mbt_xgb_line_graphs", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(df_eval.index, y_eval, label='Actual (2024)', color='black', marker='o')
    plt.plot(df_eval.index, preds_eval, label='Predicted (2024)', color='red', linestyle='--', marker='x')
    plt.title(f"{col}: Actual vs Predicted (2024)")
    plt.xlabel("Date")
    plt.ylabel("Passenger Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"downloads/mbt_xgb_line_graphs/{col}_actual_vs_pred_2024.png")
    plt.close()

# === Step 5: Combine historical + forecasted data ===
forecast_df = forecast_list[0]
for next_df in forecast_list[1:]:
    forecast_df = forecast_df.merge(next_df, on=['Date', 'Year', 'Month'], how='outer')

forecast_df = forecast_df.merge(future_weather, on=['Date', 'Year', 'Month'], how='left')
historical_df = df.reset_index()[['Date', 'Year', 'Month'] + carrier_cols + weather_cols]
final_df = pd.concat([historical_df, forecast_df], ignore_index=True)
final_df.sort_values(by='Date', inplace=True)

# === Step 6: Save forecast outputs ===
os.makedirs("downloads", exist_ok=True)
final_df.to_csv("downloads/mbt_xgboost_forecast_weather_time_full.csv", index=False)
pd.DataFrame(metrics).to_csv("downloads/mbt_xgboost_forecast_metrics.csv", index=False)
pd.DataFrame(actual_vs_predicted_rows).to_csv("downloads/mbt_xgboost_actual_vs_predicted_2024.csv", index=False)

print("Forecasting complete!")

# === Step 7: KMeans Clustering ===
print("Running KMeans clustering...")

carrier_data = df[carrier_cols].copy()
scaler = StandardScaler()
normalized = scaler.fit_transform(carrier_data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(normalized)
df['Cluster_Label'] = kmeans.labels_

df.reset_index()[['Date', 'Cluster_Label']].to_csv("downloads/mbt_carrier_clusters.csv", index=False)
print("Clustering output → downloads/mbt_carrier_clusters.csv")

# === Step 8: Random Forest Classification for High Traffic Detection ===
print("Running Random Forest classification...")

df['Total_Passengers'] = df[carrier_cols].sum(axis=1)
df['High_Traffic'] = (df['Total_Passengers'] >= df['Total_Passengers'].quantile(0.75)).astype(int)

X = df[['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'Year', 'Month']]
y = df['High_Traffic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Class distribution in test set:")
print(pd.Series(y_test).value_counts())

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

report = classification_report(y_test, preds, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv("downloads/mbt_high_traffic_classification_report.csv")
print("Classification report → downloads/mbt_high_traffic_classification_report.csv")







