import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

# Define the correct column names
column_names = ["Lane", "Auto", "Small_trucks", "large_trucks", "busses",
                "date", "facility", "time", "AWND", "PRCP", "SNOW", "SNWD",
                "TMAX_1", "TMAX_2", "TMIN"]

# Function to reshape one dataset
def reshape_to_forecast_format(df):
    # Create total_vehicles
    df["total_vehicles"] = df[["Auto", "Small_trucks", "large_trucks", "busses"]].sum(axis=1)

    # Fix time safely
    df["time"] = pd.to_numeric(df["time"], errors="coerce")  # Coerce invalid to NaN
    df = df.dropna(subset=["time"])  # Drop rows where time is invalid
    df["time"] = df["time"].astype(int).astype(str).str.zfill(4)
    df["time"] = df["time"].str[:2] + ":" + df["time"].str[2:]

    # Parse date and time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour

    # Create dayofweek and day_type
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day_type"] = np.where(df["dayofweek"].isin([5, 6]), "Weekend", "Weekday")

    # Select final columns
    final = df[["date", "time", "facility", "total_vehicles", "hour", "dayofweek", "day_type"]]
    return final

# Step 1: Load each file separately
df1 = pd.read_csv("Traffic_Data_Weather1.csv", header=None, names=column_names)
df2 = pd.read_csv("Traffic_Data_Weather2.csv", header=None, names=column_names)
df3 = pd.read_csv("Traffic_Data_Weather2.5.csv", header=None, names=column_names)
df4 = pd.read_csv("Traffic_Data_Weather3.csv", header=None, names=column_names)

# Step 2: Reshape each individually
df1_reshaped = reshape_to_forecast_format(df1)
df2_reshaped = reshape_to_forecast_format(df2)
df3_reshaped = reshape_to_forecast_format(df3)
df4_reshaped = reshape_to_forecast_format(df4)

# Step 3: Combine all reshaped datasets
final_forecast_df = pd.concat([df1_reshaped, df2_reshaped, df3_reshaped, df4_reshaped], ignore_index=True)

# Step 4: Save and download
final_forecast_df.to_csv("Traffic_dataset_combined.csv", index=False)
print(" Final reshaped and combined dataset saved as 'Traffic_dataset_combined.csv'.")



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Convert numeric columns safely
num_cols = ["Auto", "Small_trucks", "large_trucks", "busses",
            "AWND", "PRCP", "SNOW", "SNWD", "TMAX_2", "TMIN"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Fix time column and create timestamp
df["time"] = df["time"].apply(lambda x: f"{int(x):04d}")
df["time_str"] = df["time"].str[:2] + ":" + df["time"].str[2:]
df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time_str"], errors="coerce")

# Total vehicles
df["total_vehicles"] = df[["Auto", "Small_trucks", "large_trucks", "busses"]].sum(axis=1)

# Extract date/time parts
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# KMeans clustering to classify traffic levels
kmeans_data = df[["total_vehicles", "hour"]].dropna()
scaled = (kmeans_data - kmeans_data.mean()) / kmeans_data.std()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[kmeans_data.index, "traffic_level"] = kmeans.fit_predict(scaled)
df["traffic_level"] = df["traffic_level"].map({0: "Free", 1: "Moderate", 2: "Peak"})

print(" Feature engineering completed.")
df.head()

# Group by facility and timestamp
grouped = df.groupby(["facility", "timestamp"]).agg({
    "Auto": "sum", "Small_trucks": "sum", "large_trucks": "sum", "busses": "sum",
    "total_vehicles": "sum", "AWND": "mean", "PRCP": "mean", "SNOW": "mean",
    "SNWD": "mean", "TMAX_2": "mean", "TMIN": "mean"
}).reset_index()

# Extract datetime parts
grouped["date"] = grouped["timestamp"].dt.date
grouped["hour"] = grouped["timestamp"].dt.hour
grouped["year"] = grouped["timestamp"].dt.year
grouped["month"] = grouped["timestamp"].dt.month
grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek
grouped["time"] = grouped["timestamp"].dt.strftime("%H:%M")

#  Explicit split:
# 2015–2023 → training
# 2024 → optional validation (not used for forecasting, but for model testing if needed)
train_df = grouped[grouped["year"] <= 2023].copy()
valid_df = grouped[grouped["year"] == 2024].copy()

print(" Train years:", train_df["year"].unique())
print(" Train shape:", train_df.shape)
print(" Validation shape:", valid_df.shape)

#  You don't need a test_df — Chunk 4 generates 2025–2030 predictions manually

# Add lag features (optional for modeling)
grouped.sort_values(["facility", "timestamp"], inplace=True)
grouped["lag_1h"] = grouped.groupby("facility")["total_vehicles"].shift(1)
grouped["lag_24h"] = grouped.groupby("facility")["total_vehicles"].shift(24)
grouped["rolling_mean_24h"] = grouped.groupby("facility")["total_vehicles"].transform(lambda x: x.rolling(24).mean())
grouped.dropna(subset=["lag_1h", "lag_24h", "rolling_mean_24h"], inplace=True)

print(" Grouped data ready with lag features.")


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

forecast_all = []

xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

facilities = train_df["facility"].unique()

# Forecast loop
for facility in facilities:
    print(f" Training XGBoost for facility: {facility}")
    facility_data = train_df[train_df["facility"] == facility].copy()

    # Outlier removal
    q1 = facility_data["total_vehicles"].quantile(0.25)
    q3 = facility_data["total_vehicles"].quantile(0.75)
    iqr = q3 - q1
    facility_data = facility_data[
        (facility_data["total_vehicles"] >= (q1 - 1.5 * iqr)) &
        (facility_data["total_vehicles"] <= (q3 + 1.5 * iqr))
    ]

    # Feature engineering
    facility_data["day_of_week"] = pd.to_datetime(facility_data["date"]).dt.dayofweek
    facility_data["month"] = pd.to_datetime(facility_data["date"]).dt.month

    features = ["hour", "AWND", "PRCP", "SNOW", "SNWD", "TMAX_2", "TMIN", "day_of_week", "month"]
    X = facility_data[features]
    y = np.log1p(facility_data["total_vehicles"])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f" RMSE for {facility}: {rmse:.2f}")

    # Prepare future (test) data
    future = train_df[train_df["facility"] == facility].copy()
    future["day_of_week"] = pd.to_datetime
future.head(25)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

forecast_all = []

#  Enhanced parameters for better performance
xgb_params = {
    "n_estimators": 300,  # more trees = better learning
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

facilities = train_df["facility"].unique()

#  Future range: Jan 1, 2025 – Dec 31, 2030 (hourly)
future_dates = pd.date_range(start="2025-01-01", end="2030-12-31 23:00", freq="H")

for facility in facilities:
    print(f" Training XGBoost for facility: {facility}")
    facility_data = train_df[train_df["facility"] == facility].copy()

    #  Outlier removal
    q1 = facility_data["total_vehicles"].quantile(0.25)
    q3 = facility_data["total_vehicles"].quantile(0.75)
    iqr = q3 - q1
    facility_data = facility_data[
        (facility_data["total_vehicles"] >= (q1 - 1.5 * iqr)) &
        (facility_data["total_vehicles"] <= (q3 + 1.5 * iqr))
    ]

    if facility_data.empty:
        print(f"️ Skipping {facility} — no training data after outlier removal.")
        continue

    #  Feature engineering
    facility_data["day_of_week"] = pd.to_datetime(facility_data["date"]).dt.dayofweek
    facility_data["month"] = pd.to_datetime(facility_data["date"]).dt.month
    facility_data["year"] = pd.to_datetime(facility_data["date"]).dt.year

    features = ["hour", "AWND", "PRCP", "SNOW", "SNWD", "TMAX_2", "TMIN",
                "day_of_week", "month", "year"]
    X = facility_data[features]
    y = np.log1p(facility_data["total_vehicles"])

    #  Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    #  Evaluate
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f" RMSE for {facility}: {rmse:.2f}")

    #  Generate future DataFrame
    future_df = pd.DataFrame({
        "timestamp": future_dates,
        "date": future_dates.date,
        "time": future_dates.strftime("%H:%M"),
        "hour": future_dates.hour,
        "day_of_week": future_dates.dayofweek,
        "month": future_dates.month,
        "year": future_dates.year,
        "facility": facility
    })

    #  Simulate seasonal weather by month from training data
    monthly_weather = facility_data.groupby("month")[["AWND", "PRCP", "SNOW", "SNWD", "TMAX_2", "TMIN"]].mean()
    future_df = future_df.merge(monthly_weather, on="month", how="left")

    #  Predict and convert back from log scale
    future_df["total_vehicles"] = np.expm1(model.predict(future_df[features])).round().astype(int)

    forecast_all.append(future_df[["date", "time", "facility", "total_vehicles"]])

#  Combine forecasts
if forecast_all:
    forecast_df = pd.concat(forecast_all, ignore_index=True)
    print(" Forecasting complete. Combined forecast shape:", forecast_df.shape)
else:
    forecast_df = pd.DataFrame(columns=["date", "time", "facility", "total_vehicles"])
    print(" No forecast generated. `forecast_df` is empty.")

#  Preview
forecast_df.head(24)


forecast_df = pd.concat(forecast_all, ignore_index=True)
print(" Forecasting complete. Combined forecast shape:", forecast_df.shape)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#  Step 1: Parse datetime & create weekday/weekend labels
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
forecast_df["hour"] = pd.to_datetime(forecast_df["time"], format="%H:%M").dt.hour
forecast_df["dayofweek"] = forecast_df["date"].dt.dayofweek
forecast_df["day_type"] = np.where(forecast_df["dayofweek"].isin([5, 6]), "Weekend", "Weekday")

#  Step 2: Average traffic per hour/facility/day_type
busiest_by_hour = (
    forecast_df.groupby(["facility", "day_type", "hour"], as_index=False)["total_vehicles"]
    .mean()
)

#  Step 3: Quantile-based traffic level classification
traffic_level_list = []

for (fac, day_type), group in busiest_by_hour.groupby(["facility", "day_type"]):
    if group.empty or group["total_vehicles"].nunique() < 3:
        continue  # Skip groups with no variation

    q = group["total_vehicles"].quantile([1/3, 2/3])
    group["traffic_level"] = np.where(
        group["total_vehicles"] <= q.iloc[0], "Free",
        np.where(group["total_vehicles"] <= q.iloc[1], "Moderate", "Peak")
    )
    traffic_level_list.append(group)

#  Step 4: Combine and visualize
if traffic_level_list:
    busiest_by_hour = pd.concat(traffic_level_list, ignore_index=True)

    # ️ Print Top 3 busiest hours for each facility and day type
    for facility in busiest_by_hour["facility"].unique():
        print(f" Busiest Hours for {facility} by Day Type:")
        sub = busiest_by_hour[busiest_by_hour["facility"] == facility]
        for day_type in ["Weekday", "Weekend"]:
            subset = sub[sub["day_type"] == day_type]
            if not subset.empty:
                top_hours = subset.sort_values("total_vehicles", ascending=False).head(3)
                print(f"   {day_type}:")
                print(top_hours[["hour", "total_vehicles", "traffic_level"]].to_string(index=False))

    #  Bar chart grid with Seaborn
    sns.set(style="whitegrid")
    g = sns.catplot(
        data=busiest_by_hour,
        x="hour", y="total_vehicles", hue="traffic_level",
        col="facility", row="day_type",
        kind="bar", palette="Set2", height=4, aspect=1.4
    )

    g.set_titles("{row_name} • {col_name}")
    g.set_axis_labels("Hour of Day", "Avg Total Vehicles")
    g._legend.set_title("Traffic Level")
    plt.suptitle(" Forecasted Busiest Hours by Day Type and Facility (2025–2030)", y=1.03, fontsize=16)
    plt.tight_layout()
    plt.show()

else:
    print("️ No data to visualize busiest hours. Check forecast_df or model predictions.")


print(forecast_df[["date", "time", "facility", "total_vehicles"]].head())
print(forecast_df["facility"].unique())
print(forecast_df["date"].min(), "to", forecast_df["date"].max())
print(forecast_df["day_type"].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

#  Step 1: Add datetime and extract time-based aggregations
forecast_df["datetime"] = pd.to_datetime(forecast_df["date"].astype(str) + " " + forecast_df["time"])
forecast_df["week"] = forecast_df["datetime"].dt.to_period("W").apply(lambda r: r.start_time)
forecast_df["month"] = forecast_df["datetime"].dt.to_period("M").apply(lambda r: r.start_time)
forecast_df["year"] = forecast_df["datetime"].dt.year

sns.set(style="whitegrid")

#  WEEKLY AVERAGE PLOT
weekly = forecast_df.groupby(["facility", "week"], as_index=False)["total_vehicles"].mean()

plt.figure(figsize=(14, 5))
for fac in weekly["facility"].unique():
    subset = weekly[weekly["facility"] == fac]
    plt.plot(subset["week"], subset["total_vehicles"], label=fac)
plt.title(" Weekly Average Traffic Volume (2025–2030)", fontsize=14)
plt.xlabel("Week")
plt.ylabel("Avg Vehicles")
plt.legend(title="Facility", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ️ MONTHLY AVERAGE PLOT
monthly = forecast_df.groupby(["facility", "month"], as_index=False)["total_vehicles"].mean()

plt.figure(figsize=(14, 5))
for fac in monthly["facility"].unique():
    subset = monthly[monthly["facility"] == fac]
    plt.plot(subset["month"], subset["total_vehicles"], label=fac)
plt.title("️ Monthly Average Traffic Volume (2025–2030)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Avg Vehicles")
plt.legend(title="Facility", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

#  YEARLY AVERAGE PLOT
yearly = forecast_df.groupby(["facility", "year"], as_index=False)["total_vehicles"].mean()

plt.figure(figsize=(10, 5))
for fac in yearly["facility"].unique():
    subset = yearly[yearly["facility"] == fac]
    plt.plot(subset["year"], subset["total_vehicles"], marker="o", label=fac)
plt.title(" Yearly Average Traffic Volume (2025–2030)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Avg Vehicles")
plt.xticks(sorted(yearly["year"].unique()))
plt.legend(title="Facility", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#  Ensure datetime and breakdown columns exist
forecast_df["datetime"] = pd.to_datetime(forecast_df["date"].astype(str) + " " + forecast_df["time"])
forecast_df["week"] = forecast_df["datetime"].dt.to_period("W").apply(lambda r: r.start_time)
forecast_df["year"] = forecast_df["datetime"].dt.year

# ===  1. Weekly Traffic with Holiday Dips Highlighted ===
weekly = forecast_df.groupby(["facility", "week"], as_index=False)["total_vehicles"].mean()

plt.figure(figsize=(14, 5))
for fac in weekly["facility"].unique():
    subset = weekly[weekly["facility"] == fac]
    plt.plot(subset["week"], subset["total_vehicles"], label=fac)

#  Highlight New Year (Jan 1) and Christmas (Dec 25) for each year
for year in range(2025, 2031):
    for holiday in [f"{year}-01-01", f"{year}-12-25"]:
        plt.axvline(pd.to_datetime(holiday), color="gray", linestyle="--", alpha=0.3)

plt.title(" Weekly Avg Traffic Volume (2025–2030) with Holiday Dips", fontsize=14)
plt.xlabel("Week")
plt.ylabel("Avg Vehicles")
plt.legend(title="Facility", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.7)
plt.tight_layout()
plt.show()


# ===  2. Yearly Average Traffic by Facility ===
yearly = forecast_df.groupby(["facility", "year"], as_index=False)["total_vehicles"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly, x="year", y="total_vehicles", hue="facility", marker="o")

plt.title(" Yearly Avg Traffic Volume by Facility (2025–2030)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Avg Vehicles")
plt.xticks(sorted(yearly["year"].unique()))
plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.7)
plt.legend(title="Facility", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()


# Save all important columns to CSV
forecast_df[["date", "time", "facility", "total_vehicles", "hour", "dayofweek", "day_type"]].to_csv("forecast_traffic_2025_2030_full.csv", index=False)
print(" File saved as 'forecast_traffic_2025_2030_full.csv'")

# Download it



# question 5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Prepare 2019 data from historical grouped
grouped["year"] = grouped["timestamp"].dt.year
df_2019 = grouped[grouped["year"] == 2019].copy()
avg_2019 = df_2019.groupby("facility", as_index=False)["total_vehicles"].mean().rename(columns={"total_vehicles": "avg_2019"})

#  Prepare 2025 forecasted data
forecast_df["year"] = pd.to_datetime(forecast_df["date"]).dt.year
df_2025 = forecast_df[forecast_df["year"] == 2025].copy()
avg_2025 = df_2025.groupby("facility", as_index=False)["total_vehicles"].mean().rename(columns={"total_vehicles": "avg_2025"})

#  Merge and compare
avg_by_year = avg_2019.merge(avg_2025, on="facility", how="inner")
avg_by_year["percent_change"] = ((avg_by_year["avg_2025"] - avg_by_year["avg_2019"]) / avg_by_year["avg_2019"]) * 100
avg_by_year = avg_by_year.round(2)

#  Summary Print
print(" === 2025 Forecast vs 2019 Pre-COVID Comparison ===")
for _, row in avg_by_year.iterrows():
    fac = row["facility"]
    change = row["percent_change"]
    trend = " Increase" if change > 1 else " Decrease" if change < -1 else "No Significant Change Similar"
    print(f"{fac:>15}: 2019 = {row['avg_2019']:.0f}, 2025 = {row['avg_2025']:.0f}, Change = {change:+.2f}% → {trend}")

#  Optional: Visualize
plt.figure(figsize=(10, 5))
sns.barplot(data=avg_by_year, x="facility", y="percent_change", palette="coolwarm")
plt.axhline(0, color="gray", linestyle="--")
plt.title(" % Change in Forecasted Traffic (2025 vs 2019)")
plt.ylabel("% Change from 2019")
plt.xlabel("Facility")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#  Step 1: Extract monthly data for 2019 (grouped) and 2025 (forecast_df)
df_2019 = grouped[grouped["timestamp"].dt.year == 2019].copy()
df_2019["month"] = df_2019["timestamp"].dt.month
df_2019["year"] = 2019

df_2025 = forecast_df[pd.to_datetime(forecast_df["date"]).dt.year == 2025].copy()
df_2025["month"] = pd.to_datetime(df_2025["date"]).dt.month
df_2025["year"] = 2025

#  Step 2: Combine both for comparison
compare_df = pd.concat([
    df_2019[["facility", "month", "year", "total_vehicles"]],
    df_2025[["facility", "month", "year", "total_vehicles"]]
], ignore_index=True)

#  Step 3: Average by facility, month, and year
monthly_comp = compare_df.groupby(["facility", "year", "month"], as_index=False)["total_vehicles"].mean()

#  Step 4: Plot per facility
for fac in monthly_comp["facility"].unique():
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=monthly_comp[monthly_comp["facility"] == fac],
        x="month", y="total_vehicles", hue="year", marker="o"
    )
    plt.title(f" Monthly Traffic Comparison (2025 vs 2019) – {fac}")
    plt.xlabel("Month")
    plt.ylabel("Avg Vehicles")
    plt.xticks(range(1, 13))
    plt.legend(title="Year")
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np

#  Extract 2019 data from actuals
df_2019 = grouped[grouped["timestamp"].dt.year == 2019].copy()
df_2019["year"] = 2019
df_2019["month"] = df_2019["timestamp"].dt.month
df_2019["week"] = df_2019["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
df_2019["dayofweek"] = df_2019["timestamp"].dt.dayofweek
df_2019["hour"] = df_2019["timestamp"].dt.hour
df_2019["day_type"] = np.where(df_2019["dayofweek"].isin([5, 6]), "Weekend", "Weekday")

#  Extract 2025 data from forecast
df_2025 = forecast_df[pd.to_datetime(forecast_df["date"]).dt.year == 2025].copy()
df_2025["timestamp"] = pd.to_datetime(df_2025["date"].astype(str) + " " + forecast_df["time"])
df_2025["year"] = 2025
df_2025["month"] = df_2025["timestamp"].dt.month
df_2025["week"] = df_2025["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
df_2025["dayofweek"] = df_2025["timestamp"].dt.dayofweek
df_2025["hour"] = df_2025["timestamp"].dt.hour
df_2025["day_type"] = np.where(df_2025["dayofweek"].isin([5, 6]), "Weekend", "Weekday")

#  Combine
compare_df = pd.concat([df_2019, df_2025], ignore_index=True)

# === Yearly Average
year_avg = compare_df.groupby(["facility", "year"], as_index=False)["total_vehicles"].mean()
pivot_year = year_avg.pivot(index="facility", columns="year", values="total_vehicles").reset_index()
pivot_year["%_change"] = ((pivot_year[2025] - pivot_year[2019]) / pivot_year[2019] * 100).round(2)

# === Monthly Patterns
monthly = compare_df.groupby(["facility", "year", "month"], as_index=False)["total_vehicles"].mean()

# === Weekly Patterns
weekly = compare_df.groupby(["facility", "year", "week"], as_index=False)["total_vehicles"].mean()

# === Hourly Patterns (Weekday vs Weekend)
hourly = compare_df.groupby(["facility", "year", "day_type", "hour"], as_index=False)["total_vehicles"].mean()

#  PRINT COMPARISON SUMMARY
print(" === 2025 vs 2019 TRAFFIC COMPARISON SUMMARY === ")

for idx, row in pivot_year.iterrows():
    fac = row["facility"]
    y2019 = row[2019]
    y2025 = row[2025]
    change = row["%_change"]
    trend = " Increase" if change > 1 else " Decrease" if change < -1 else "No Significant Change Similar"

    print(f" {fac}")
    print(f"   • Avg in 2019: {y2019:.0f}")
    print(f"   • Avg in 2025: {y2025:.0f}")
    print(f"   • % Change   : {change:+.2f}% — {trend}")

    # Monthly summer trend
    monthly_fac = monthly[(monthly["facility"] == fac)]
    summer_2019 = monthly_fac[(monthly_fac["year"] == 2019) & (monthly_fac["month"].isin([6, 7, 8]))]["total_vehicles"].mean()
    summer_2025 = monthly_fac[(monthly_fac["year"] == 2025) & (monthly_fac["month"].isin([6, 7, 8]))]["total_vehicles"].mean()

    if not np.isnan(summer_2019) and not np.isnan(summer_2025):
        summer_diff = ((summer_2025 - summer_2019) / summer_2019) * 100
        summer_trend = "Increase" if summer_diff > 0 else "Decrease" if summer_diff < 0 else "-"
        print(f"   • Summer Trend (Jun–Aug): {summer_trend} {summer_diff:.1f}%")

    # Rush hour pattern
    wk_2019 = hourly[(hourly["facility"] == fac) & (hourly["year"] == 2019) & (hourly["day_type"] == "Weekday")]["total_vehicles"].max()
    wk_2025 = hourly[(hourly["facility"] == fac) & (hourly["year"] == 2025) & (hourly["day_type"] == "Weekday")]["total_vehicles"].max()

    if not np.isnan(wk_2019) and not np.isnan(wk_2025):
        peak_change = ((wk_2025 - wk_2019) / wk_2019) * 100
        rush_trend = "Flattened" if abs(peak_change) < 5 else "Sharpened" if peak_change > 5 else "Reduced"
        print(f"   • Rush Hour Intensity: {rush_trend} ({peak_change:+.1f}%)")

    print()