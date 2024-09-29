import pandas as pd

# Load the data
data = pd.read_csv("lite_golden_search_results.csv", sep=";")

print(data.head())
