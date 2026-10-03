import pandas as pd
import numpy as np
from scipy import stats

scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

print("\na)\n")
print(scores.describe())

print("\nb)\n")
clean_scores = scores.dropna()
print("Skewness:", stats.skew(clean_scores))
print("Kurtosis:", stats.kurtosis(clean_scores))

print("\nc)\n")
print("Missing values:", scores.isnull().sum())

print("\nd)\n")
filled_scores = scores.fillna(scores.mean())
print("Filled scores:")
print(filled_scores)