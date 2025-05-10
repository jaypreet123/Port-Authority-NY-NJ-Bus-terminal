# In[1]:


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

# === Step 1: Load dataset ===
df = pd.read_csv("/Users/jay/Desktop/Project-goalcode/Data-sets/Goal-3/gwbbs_project_goal3.1-new.csv")
df.columns = df.columns.str.strip()

carrier_cols = [
    'Hudson_Transit', 'Rockland', 'NJ_Transit', 'Spanish',
    'Greyhound_LD', 'OurBus_LD', 'Saddle_River', 'Saddle_River_LD',
    'Vanessa', 'Total_GWBBS', 'Total_Buses', 'Total_Passengers'
]
weather_cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2))
df['Month'] = df['Date'].dt.month
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# === Step 2: Clean missing or zero values ===
for col in weather_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df.groupby('Month')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

for col in carrier_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df.groupby('Month')[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

# === Step 3: Prepare future data (2025–2030) ===
np.random.seed(42)
future_dates = pd.date_range("2025-01-01", "2030-12-01", freq='MS')
monthly_weather_avg = df.groupby('Month')[weather_cols].mean().reset_index()

future_weather = pd.DataFrame({
    'Date': future_dates,
    'Month': future_dates.month,
    'Year': future_dates.year
})
future_weather = future_weather.merge(monthly_weather_avg, on='Month', how='left')

# Add variation to weather
for col in weather_cols:
    std_dev = df[col].std() * 0.05
    noise = np.random.normal(0, std_dev, size=len(future_weather))
    future_weather[col] += noise
    future_weather[col] = future_weather[col].clip(lower=0)

# === Step 4: Train/Test Split ===
df_eval = df[df['Year'] == 2024]
df_train = df[df['Year'] < 2024]

forecast_list = []
metrics = []
actual_vs_predicted_rows = []

# === Step 5: Forecast each carrier ===
for col in carrier_cols:
    print(f"🚍 Forecasting {col}...")

    X_train = df_train[weather_cols + ['Year', 'Month']]
    y_train = df_train[col]
    X_eval = df_eval[weather_cols + ['Year', 'Month']]
    y_eval = df_eval[col]
    X_future = future_weather[weather_cols + ['Year', 'Month']]

    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [4],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    preds_eval = best_model.predict(X_eval)
    preds_future = best_model.predict(X_future)

    # === Add sinusoidal variation + noise ===
    months_since_2025 = np.arange(len(preds_future))
    sin_wave = 1 + 0.05 * np.sin(2 * np.pi * months_since_2025 / 12)
    noise = np.random.normal(1.0, 0.05, size=len(preds_future))
    preds_future *= sin_wave * noise

    # Clip and upscale
    min_val = df[df[col] > 0][col].quantile(0.05)
    max_val = df[col].quantile(0.95)
    preds_future = np.clip(preds_future, min_val, max_val)
    preds_future *= 1.1

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

    os.makedirs("downloads/xgb_line_graphs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(df_eval.index, y_eval, label='Actual (2024)', color='black')
    plt.plot(df_eval.index, preds_eval, label='Predicted (2024)', color='red', linestyle='--')
    plt.title(f"{col}: Actual vs Predicted (2024)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"downloads/xgb_line_graphs/{col}_actual_vs_pred_2024.png")
    plt.close()

# === Step 6: Combine forecast with history ===
forecast_df = forecast_list[0]
for next_df in forecast_list[1:]:
    forecast_df = forecast_df.merge(next_df, on=['Date', 'Year', 'Month'], how='outer')

forecast_df = forecast_df.merge(future_weather, on=['Date', 'Year', 'Month'], how='left')
historical_df = df.reset_index()[['Date', 'Year', 'Month'] + carrier_cols + weather_cols]
final_df = pd.concat([historical_df, forecast_df], ignore_index=True)
final_df.sort_values(by='Date', inplace=True)

# === Step 7: Export ===
os.makedirs("downloads", exist_ok=True)
final_df.to_csv("downloads/xgboost_forecast_weather_time_full.csv", index=False)
pd.DataFrame(metrics).to_csv("downloads/xgboost_forecast_metrics.csv", index=False)
pd.DataFrame(actual_vs_predicted_rows).to_csv("downloads/xgboost_actual_vs_predicted_2024.csv", index=False)

# === Step 8: KMeans Clustering ===
scaler = StandardScaler()
normalized = scaler.fit_transform(df[carrier_cols])
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster_Label'] = kmeans.fit_predict(normalized)
df.reset_index()[['Date', 'Cluster_Label']].to_csv("downloads/carrier_clusters.csv", index=False)

# === Step 9: Random Forest Classifier ===
df['High_Traffic'] = (df['Total_Passengers'] >= df['Total_Passengers'].quantile(0.75)).astype(int)
X = df[['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'Year', 'Month']]
y = df['High_Traffic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
report = classification_report(y_test, clf.predict(X_test), output_dict=True)
pd.DataFrame(report).transpose().to_csv("downloads/high_traffic_classification_report.csv")

print("Data ready for Power BI")





