import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('advertising.csv')

# Separate features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']] 
y = data['Sales']

# Histograms of numerical columns before transformation
data.hist(figsize=(10, 6))
plt.suptitle("Histograms of Numerical Columns (Before Transformation)")
plt.show()

# Logarithm transformation of features
X_log = np.log1p(X)

# Histograms of numerical columns after transformation
X_log.hist(figsize=(10, 6))
plt.suptitle("Histograms of Numerical Columns (After Transformation)")
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optionally, visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
