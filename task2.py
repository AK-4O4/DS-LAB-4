import pandas as pd
import numpy as np
from scipy.stats import zscore

data = {
    "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}
df = pd.DataFrame(data)

print("\na)\n")
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())
print("After median fill:")
print(df[["Day", "Revenue"]])

print("\nb)\n")
revenue = df["Revenue"]
df["Revenue_MinMax"] = (revenue - revenue.min()) / (revenue.max() - revenue.min())
df["Revenue_Zscore"] = zscore(revenue)

print("\nc)\n")
Q1 = revenue.quantile(0.25)
Q3 = revenue.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df["Outlier"] = (revenue < lower_bound) | (revenue > upper_bound)
print("IQR bounds:", lower_bound, "-", upper_bound)
print("IQR outliers detected on day(s):", df[df["Outlier"]]["Day"].tolist())

print("\nd)\n")
print(df[["Day", "Revenue", "Revenue_MinMax", "Revenue_Zscore", "Outlier"]])