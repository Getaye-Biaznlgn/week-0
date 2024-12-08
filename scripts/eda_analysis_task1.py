import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from windrose import WindroseAxes


os.system("git checkout -b task-1")

data_path = "benin-malanville.csv"  
data = pd.read_csv(data_path)


summary_stats = data.describe()
print("Summary Statistics:\n", summary_stats)

missing_values = data.isnull().sum()
outliers = data.apply(lambda x: np.sum(zscore(x) > 3) if x.dtype in [np.float64, np.int64] else 0)
print("Missing Values:\n", missing_values)
print("Outliers:\n", outliers)

columns_with_outliers = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
for col in columns_with_outliers:
    if col in data.columns:
        z_scores = zscore(data[col].dropna())
        data = data.loc[abs(z_scores) <= 3]

if 'DateTime' in data.columns:
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)
    monthly_avg = data[['GHI', 'DNI', 'DHI', 'Tamb']].resample('M').mean()
    monthly_avg.plot(title='Monthly Averages')
    plt.show()

if 'Cleaning' in data.columns:
    cleaning_impact = data.groupby('Cleaning')[['ModA', 'ModB']].mean()
    cleaning_impact.plot(kind='bar', title='Impact of Cleaning on ModA and ModB')
    plt.show()

correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust']])
plt.show()

if 'WD' in data.columns:
    ax = WindroseAxes.from_ax()
    ax.bar(data['WD'], data['WS'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.show()

# Temperature Analysis
if 'RH' in data.columns:
    sns.scatterplot(x=data['RH'], y=data['Tamb'], hue=data['GHI'], size=data['BP'], sizes=(20, 200))
    plt.title("Temperature vs RH with GHI and BP")
    plt.show()

# Histograms
for col in ['GHI', 'DNI', 'DHI', 'WS', 'Tamb']:
    if col in data.columns:
        plt.hist(data[col].dropna(), bins=30, alpha=0.7, label=col)
        plt.title(f"Histogram for {col}")
        plt.show()

# Z-Score Analysis
for col in columns_with_outliers:
    if col in data.columns:
        data[f'{col}_zscore'] = zscore(data[col].fillna(data[col].mean()))

# Bubble chart example
sns.scatterplot(x=data['GHI'], y=data['Tamb'], size=data['WS'], hue=data['RH'], sizes=(40, 400), alpha=0.5)
plt.title("Bubble Chart: GHI vs Tamb vs WS")
plt.show()


data.fillna(data.mean(), inplace=True)
data.drop(columns=['Comments'], errors='ignore', inplace=True)

cleaned_data_path = "cleaned_data.csv"
data.to_csv(cleaned_data_path, index=False)


