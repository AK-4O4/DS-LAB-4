import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

data = {
    "Hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "Temperature": [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23,
                    22, 21, 23, 24, 25]
}
df = pd.DataFrame(data)
df["Temp_raw"] = df["Temperature"]

print("\na) Stats BEFORE cleaning:\n")
print(df["Temperature"].describe())
temp_no_nan = df["Temperature"].dropna()
print("Skewness:", stats.skew(temp_no_nan))
print("Kurtosis:", stats.kurtosis(temp_no_nan))
mean_before = df["Temperature"].mean()
std_before = df["Temperature"].std()

print("\nb) After interpolation:\n")
df["Temperature"] = df["Temperature"].interpolate(method="linear")
print(df["Temperature"].isnull().sum())

print("\nc) Z-score(>2) outlier detected at Hour: \n")
df["abs_zscore"] = pd.Series(zscore(df["Temperature"])).abs()
median_value = df["Temperature"].median()
outlier_mask = df["abs_zscore"] > 2
for i in df[outlier_mask].index:
    print("Z-score(>2) outlier detected at Hour:", df.loc[i, "Hour"], "(value:", df.loc[i, "Temperature"], ")")
print("Replaced with column median:", median_value)
df.loc[outlier_mask, "Temperature"] = median_value

print("\nd) Z-score standardization to the final cleaned Temperature: \n")
df["Temp_clean"] = df["Temperature"]
df["Temp_Zscore"] = zscore(df["Temp_clean"])
print("Before/After comparison:")
print(df[["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]])

mean_after = df["Temp_clean"].mean()
std_after = df["Temp_clean"].std()
print("Stats AFTER cleaning:")
print("Mean:", mean_after, "Std:", std_after)
print("BEFORE: Mean", mean_before, "Std", std_before)