import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Read the data from the CSV file
df = pd.read_csv('admissions_data.csv')

# Display the first few rows of the dataframe
print(df.head())

# Summary statistics of the dataframe
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualizing the distribution of GRE Score
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='GRE Score', kde=True, bins=20)
plt.title('Distribution of GRE Scores')
plt.xlabel('GRE Score')
plt.ylabel('Frequency')
plt.show()

# Visualizing the relationship between CGPA and Chance of Admit
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CGPA', y='Chance of Admit ')
plt.title('CGPA vs Chance of Admit')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.show()

# Visualizing the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Splitting the data into features and target variable
X = df.drop(columns=['Chance of Admit '])
y = df['Chance of Admit ']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

