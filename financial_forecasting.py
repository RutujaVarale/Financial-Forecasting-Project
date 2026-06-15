import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load Excel file
df = pd.read_excel("sales.xlsx")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Set Date as index
df.set_index("Date", inplace=True)

print("Dataset Loaded Successfully")
print(df.head())

# Historical Revenue Plot
plt.figure(figsize=(10, 5))
plt.plot(df["Revenue"], marker="o")
plt.title("Historical Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
plt.savefig("historical_revenue.png")
plt.show()

# Build ARIMA Model
model = ARIMA(df["Revenue"], order=(2, 1, 2))
model_fit = model.fit()

# Forecast Next 6 Months
forecast = model_fit.forecast(steps=6)

print("\nForecasted Revenue:")
print(forecast)

# Future Dates
future_dates = pd.date_range(
    start=df.index[-1],
    periods=7,
    freq="MS"
)[1:]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast.values
})

# Forecast Plot
plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df["Revenue"],
    label="Historical Revenue",
    marker="o"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    label="Forecast Revenue",
    marker="o"
)

plt.title("Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)

plt.savefig("forecast_plot.png")
plt.show()

# Save Forecast Output
forecast_df.to_csv(
    "forecast_output.csv",
    index=False
)

print("\nForecast saved to forecast_output.csv")
print("Historical graph saved to historical_revenue.png")
print("Forecast graph saved to forecast_plot.png")