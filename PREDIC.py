import pandas as pd
import numpy as np
import joblib

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "return_1", "log_return", "volatility_10"
]

def safe_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def main():
    df = pd.read_csv("PRJDS.csv")

    if len(df) < 12:
        raise ValueError("PRJDS.csv does not have enough rows. Need at least ~12 rows to compute rolling features safely.")

    model = joblib.load("model.pkl")

    print("\n=== NEXT-DAY CLOSE PREDICTOR (You enter OHLCV only) ===\n")

    o = safe_float("Enter Open: ")
    h = safe_float("Enter High: ")
    l = safe_float("Enter Low: ")
    c = safe_float("Enter Close: ")
    v = safe_float("Enter Volume: ")

    # Previous close from the last available row
    prev_close = float(df.iloc[-1]["Close"])

    # Compute features exactly like DataProcessing.py
    ma5 = pd.Series(list(df["Close"].tail(4)) + [c]).mean()     # last 4 closes + current = 5
    ma10 = pd.Series(list(df["Close"].tail(9)) + [c]).mean()    # last 9 closes + current = 10

    return_1 = (c / prev_close) - 1.0
    log_return = np.log(c / prev_close)

    # volatility_10 = std of last 9 log_returns + current log_return (window 10)
    last9_logrets = list(df["log_return"].tail(9))
    volatility_10 = pd.Series(last9_logrets + [log_return]).std()  # pandas uses ddof=1 by default (same as rolling().std())

    # Build model input in the exact order used in training
    x = np.array([[o, h, l, c, v, ma5, ma10, return_1, log_return, volatility_10]], dtype=float)

    pred = model.predict(x)[0]

    print("\nComputed features:")
    print(f"MA5: {ma5:.6f}")
    print(f"MA10: {ma10:.6f}")
    print(f"return_1: {return_1:.6f}")
    print(f"log_return: {log_return:.6f}")
    print(f"volatility_10: {volatility_10:.6f}")

    print("\nPredicted NEXT-DAY Close Price:", round(float(pred), 6))
    print("Done.\n")

if __name__ == "__main__":
    main()