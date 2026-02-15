
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Data/PRJDS.csv")
feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]

corr_matrix = df[feature_cols + ["Target"]].corr()
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".5f")
plt.title("Feature Correlation Matrix")
plt.show()