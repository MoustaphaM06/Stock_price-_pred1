import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

sgd_model = joblib.load("model.pkl")
feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]

df=pd.read_csv("Data/PRJDS.csv")
split_idx = int(0.8 * len(df))
test_df  = df.iloc[split_idx:].copy()
X_test  = test_df[feature_cols]
y_test  = test_df["Target"]

y_pred = sgd_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#print("Test MAE:", mae)
#print("Test RMSE:", rmse)
#  print("Test R2:", r2)


results = test_df[["Date", "Target"]].copy()
results["Predicted"] = y_pred
results["Error"] = abs(results["Predicted"] - results["Target"])

print(results.head(4))  #print random results if want


plt.plot(results["Date"], results["Target"], label="Actual")
plt.plot(results["Date"], results["Predicted"], label="Predicted")
plt.xlabel("Date")
plt.ylabel("Next-day Close")
plt.title("Actual vs Predicted (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


baseline_pred = X_test["Close"].values
baseline_r2 = r2_score(y_test, baseline_pred)

print("Baseline R2:", baseline_r2)