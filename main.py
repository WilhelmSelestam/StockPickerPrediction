import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_path = 'dataset.csv'

df = pd.read_csv(data_path)


# Explore the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nStatistical summary:")
print(df.describe())