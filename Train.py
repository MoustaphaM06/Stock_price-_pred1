import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df=pd.read_csv("PRJDS.csv")
feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10","return_1","log_return","volatility_10"]

split_idx = int(0.8 * len(df))

train_df = df.iloc[:split_idx].copy()

X_train = train_df[feature_cols]
y_train = train_df["Target"]




sgd_model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", SGDRegressor(
        loss="squared_error",
        learning_rate="invscaling",
        eta0=0.01,
        alpha=0.0001,          
        max_iter=20000,
        tol=0.000006,
    
    ))
])

sgd_model.fit(X_train, y_train)

joblib.dump(sgd_model, "model.pkl")



y_pred = sgd_model.predict(X_train)



r2 = r2_score(y_train, y_pred)

print (r2)


print("\n=== MODEL STRUCTURE ===")
reg = sgd_model.named_steps["reg"]
print("\nIntercept:", reg.intercept_[0])
print("\nWeights:")
for name, weight in zip(feature_cols, reg.coef_):
    print(f"{name}: {weight:.6f}")