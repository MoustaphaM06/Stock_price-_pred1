import pandas as pd

# cleaning ldata

df = pd.read_csv("HistoricalQuotes.csv")

df["Date"] = pd.to_datetime(df["Date"])


df = df.sort_values("Date").reset_index(drop=True)
for col in ["Close", "Open", "High", "Low"]:

    df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()

df["Target"] = df["Close"].shift(-1)

df = df.dropna().reset_index(drop=True)

features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]
X = df[features]
y = df["Target"]

df.to_csv("PRJDS.csv",index=False)

