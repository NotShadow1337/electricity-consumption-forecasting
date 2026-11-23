# Mini Project - Electricity Consumption Analysis and Forecasting
# Name: Shikhar Yadav
# Roll No: 25/SCA/BCOA/066

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   #saves the graphs as images instead of opening them
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# folder where all the graphs will be saved
os.makedirs("output", exist_ok=True)
np.random.seed(42)   # so that we get the same random numbers every time


# Step 1: making the dataset 
# I don't have a real meter data so I made a dataset of 3 years (daily)

dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
day_no = dates.dayofyear.values

# temperature is high in summer and low in winter
temp = 25 + 10 * np.sin(2 * np.pi * (day_no - 105) / 365) + np.random.normal(0, 1.5, len(dates))

# base usage is 14 units, if temp is more than 28 then AC is used so usage goes up
usage = 14 + np.where(temp > 28, (temp - 28) * 0.9, 0)

# on saturday and sunday people stay at home so more usage
usage = usage + np.where(dates.dayofweek >= 5, 2.5, 0)

# adding some random noise to make it look real
usage = usage + np.random.normal(0, 1.2, len(dates))

df = pd.DataFrame({"date": dates,
                   "temperature": temp.round(1),
                   "consumption": usage.round(2)})

# removing 8 random values on purpose so we can practice cleaning
df.loc[np.random.choice(len(df), 8, replace=False), "consumption"] = np.nan
df.to_csv("electricity_data.csv", index=False)


# Step 2: cleaning the data 

print("Missing values before cleaning:", df["consumption"].isna().sum())

# filling the empty places using the values near them
df["consumption"] = df["consumption"].interpolate()

print("Missing values after cleaning :", df["consumption"].isna().sum())


# Step 3: analysis 

df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek   # monday is 0 and sunday is 6
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

weekday_data = df[df["is_weekend"] == 0]
weekend_data = df[df["is_weekend"] == 1]

print("\n--- BASIC ANALYSIS ---")
print("Total records            :", len(df))
print("Average daily usage      :", round(df["consumption"].mean(), 2), "kWh")
print("Highest daily usage      :", df["consumption"].max(), "kWh")
print("Lowest daily usage       :", df["consumption"].min(), "kWh")
print("Weekday average          :", round(weekday_data["consumption"].mean(), 2), "kWh")
print("Weekend average          :", round(weekend_data["consumption"].mean(), 2), "kWh")
print("Temperature vs usage corr:", round(df["temperature"].corr(df["consumption"]), 2))

# graph 1 - daily consumption
plt.figure(figsize=(11, 4))
plt.plot(df["date"], df["consumption"])
plt.title("Daily Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("kWh")
plt.savefig("output/1_daily_consumption.png", dpi=120, bbox_inches="tight")
plt.close()

# graph 2 - average usage of each month
monthly = df.groupby("month")["consumption"].mean()
plt.figure(figsize=(8, 4))
plt.bar(monthly.index, monthly.values, color="steelblue")
plt.title("Average Consumption by Month")
plt.xlabel("Month (1 = Jan, 12 = Dec)")
plt.ylabel("kWh per day")
plt.savefig("output/2_monthly_average.png", dpi=120, bbox_inches="tight")
plt.close()

# graph 3 - temperature vs consumption
plt.figure(figsize=(7, 4.5))
plt.scatter(df["temperature"], df["consumption"], s=8, color="green")
plt.title("Temperature vs Consumption")
plt.xlabel("Temperature (deg C)")
plt.ylabel("kWh")
plt.savefig("output/3_temperature_vs_consumption.png", dpi=120, bbox_inches="tight")
plt.close()


# Step 4: getting data ready for model 

# yesterday's usage and last week's usage will help in predicting today's usage
df["yesterday"] = df["consumption"].shift(1)
df["last_week"] = df["consumption"].shift(7)
data = df.dropna()   # first few rows have no lag values so removing them

cols = ["temperature", "month", "dayofweek", "is_weekend", "yesterday", "last_week"]

# first 80% days for training and last 20% days for testing
# (not using random split because this is time series data)
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:]

X_train = train[cols]
y_train = train["consumption"]
X_test = test[cols]
y_test = test["consumption"]


# Step 5: training the models 

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("\n--- MODEL RESULTS ---")
print("Linear Regression -> MAE:", round(mean_absolute_error(y_test, lr_pred), 2),
      "| R2:", round(r2_score(y_test, lr_pred), 2))
print("Random Forest     -> MAE:", round(mean_absolute_error(y_test, rf_pred), 2),
      "| R2:", round(r2_score(y_test, rf_pred), 2))
print("(less MAE and more R2 means better model, so Random Forest is better)")

# graph 4 - actual vs predicted for last 100 days of test data
plt.figure(figsize=(11, 4))
plt.plot(test["date"].values[-100:], y_test.values[-100:], label="Actual")
plt.plot(test["date"].values[-100:], rf_pred[-100:], label="Predicted", linestyle="--", color="red")
plt.title("Actual vs Predicted Consumption (Random Forest)")
plt.xlabel("Date")
plt.ylabel("kWh")
plt.legend()
plt.savefig("output/4_actual_vs_predicted.png", dpi=120, bbox_inches="tight")
plt.close()


# Step 6: forecasting next 30 days 

# for future temperature I am taking the average of that same day from past years
avg_temp = df.groupby(df["date"].dt.dayofyear)["temperature"].mean()

history = list(df["consumption"])   # all the usage values we already know
future_dates = pd.date_range("2026-01-01", periods=30)
forecast = []

for day in future_dates:
    # making one row of input for this day
    row = pd.DataFrame({
        "temperature": [avg_temp[day.dayofyear]],
        "month": [day.month],
        "dayofweek": [day.dayofweek],
        "is_weekend": [int(day.dayofweek >= 5)],
        "yesterday": [history[-1]],
        "last_week": [history[-7]],
    })
    pred = rf.predict(row)[0]
    forecast.append(round(pred, 2))
    history.append(pred)   # adding it to history so it can be used for next day

result = pd.DataFrame({"date": future_dates, "forecast_kwh": forecast})
result.to_csv("output/forecast_next_30_days.csv", index=False)

total = sum(forecast)
print("\n--- FORECAST FOR NEXT 30 DAYS ---")
print(result.head(7).to_string(index=False))
print("...")
print("Total expected usage :", round(total, 1), "kWh")
print("Average per day      :", round(np.mean(forecast), 2), "kWh")
print("Estimated bill       : Rs", round(total * 7), "(taking Rs 7 per unit)")

# graph 5 - forecast
plt.figure(figsize=(11, 4))
plt.plot(df["date"].tail(60), df["consumption"].tail(60), label="Past 60 days")
plt.plot(future_dates, forecast, label="Next 30 days (forecast)", linestyle="--", color="red")
plt.title("30-Day Electricity Consumption Forecast")
plt.xlabel("Date")
plt.ylabel("kWh")
plt.legend()
plt.savefig("output/5_forecast.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nDone, Graphs are saved in the output folder.")