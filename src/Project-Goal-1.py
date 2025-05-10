# In[7]:


pip install pandas xgboost


# In[1]:


pip install scikit-learn


# In[2]:


pip install pandas numpy


# In[7]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# Step 1: Load and clean data
df = pd.read_csv("/Users/jay/Downloads/project_goal_1.2.csv")  # path 
df.columns = df.columns.str.strip()

# Step 2: Interpolate missing values
df[['gwbbs_Passengers_Count', 'mbt_Passengers_Count', 'mbtpd_Passengers_Count',
    'AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']] = df[[
    'gwbbs_Passengers_Count', 'mbt_Passengers_Count', 'mbtpd_Passengers_Count',
    'AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'
]].interpolate(method='linear')

# Step 3: Drop incomplete records
df.dropna(subset=['gwbbs_Passengers_Count', 'mbt_Passengers_Count', 'mbtpd_Passengers_Count'], inplace=True)

# Step 4: Calculate total passengers
df['total_passengers'] = (
    df['gwbbs_Passengers_Count'] +
    df['mbt_Passengers_Count'] +
    df['mbtpd_Passengers_Count']
)

# Step 5: Aggregate by Year, Month, Time, Facility
hourly_df = df.groupby(['Year', 'Month', 'Time', 'Facilities'], as_index=False).agg({
    'total_passengers': 'sum',
    'AWND': 'mean',
    'PRCP': 'mean',
    'SNOW': 'mean',
    'SNWD': 'mean',
    'TMAX': 'mean',
    'TMIN': 'mean'
})

# Step 6: Prepare training data (2020–2024)
train_df = hourly_df[hourly_df['Year'] <= 2024].copy()
train_df = pd.get_dummies(train_df, columns=['Facilities'])

# Step 7: Train Linear Regression model
X_train = train_df.drop(columns=['total_passengers', 'Year', 'Month'])
y_train = train_df['total_passengers']

model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Analyze weather + time impact
coefficients = pd.Series(model.coef_, index=X_train.columns)
weather_cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
print("\nWeather impact on passengers:")
print(coefficients[weather_cols].sort_values(ascending=False))

print("\nTime impact:")
print(f"Time coefficient: {coefficients['Time']}")

# Step 9: Prepare future data (2025–2030)
weather_avg = train_df.groupby('Month')[weather_cols].mean().reset_index()
facilities_encoded = [col for col in train_df.columns if col.startswith('Facilities_')]
facility_map = {f: f.replace("Facilities_", "") for f in facilities_encoded}

future_rows = []
for year in range(2025, 2031):
    for month in range(1, 13):
        for hour in sorted(df['Time'].unique()):
            for fac in facilities_encoded:
                row = {'Year': year, 'Month': month, 'Time': hour}
                row.update(weather_avg[weather_avg['Month'] == month].iloc[0].drop('Month').to_dict())
                for f in facilities_encoded:
                    row[f] = 1 if f == fac else 0
                row['Facility'] = facility_map[fac]
                future_rows.append(row)

future_df = pd.DataFrame(future_rows)

# Step 10: Predict
X_future = future_df.drop(columns=['Year', 'Month', 'Facility'])
future_df['predicted_passengers'] = model.predict(X_future)

# Step 11: Final output
result_df = future_df[['Year', 'Month', 'Time', 'Facility', 'predicted_passengers'] + weather_cols]

# Step 12: Save to Downloads folder
downloads_path = os.path.expanduser("~/Downloads")
output_file = os.path.join(downloads_path, "hourly_passenger_forecast_2025_2030.csv")
result_df.to_csv(output_file, index=False)

print(f"\nForecast complete! File saved to: {output_file}")




