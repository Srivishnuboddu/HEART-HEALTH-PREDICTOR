# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split #Splits the data into training and testing sets
from sklearn.linear_model import LogisticRegression  #Implements the Logistic Regression algorithm
import pickle

# Load the dataset
data = pd.read_csv('data/heart.csv')

# Select relevant features and the target variable
# Modify this line based on the provided column names
X = data[['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Slope', 'Ca', 'Thal']]  # Features used for prediction
y = data['Target']  # Target column

# Encode categorical features (if needed)
# Convert categorical columns to numeric if they are not already encoded
X = pd.get_dummies(X, columns=['Thal', 'Slope'], drop_first=True)  # One-hot encoding for categorical features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning occurs
model.fit(X_train, y_train)

# Save the trained model as a pickle file
model_filename = 'app/model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model training complete and saved as '{model_filename}'.")
