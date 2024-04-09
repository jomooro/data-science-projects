import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Convert categorical variables into dummy/indicator variables
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'])
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'])

# Define features and target variable
X_train = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_df['Survived']
X_test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_model.predict(X_val)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy on validation set:", accuracy)

# Make predictions on the test set
test_predictions = rf_model.predict(X_test)

# Save the predictions to a CSV file
test_df['Survived'] = test_predictions
test_df[['PassengerId', 'Survived']].to_csv('result.csv', index=False)

# Evaluate the model on the test set with ground truth labels
y_test = test_df['Survived']
test_accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy on test set:", test_accuracy)

# View the classification report on the test set
print("\nClassification Report on test set:")
print(classification_report(y_test, test_predictions))
