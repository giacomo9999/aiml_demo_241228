import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")
scaler = StandardScaler()
model = LogisticRegression()

# Separate the features (X) and the target variable (y)
X = data.drop("Outcome", axis=1)  # Features (independent variables)
y = data["Outcome"]  # Target (dependent variable)
X_scaled = scaler.fit_transform(X)  # normalize the imported data to 1(?)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

new_patient = [
    [1.5, 85, 66, 29, 0, 26.6, 0.351, 31]
]  # new row of data for a new patient - prediction 1

outcome_zero_patient = [
    [8, 99, 84, 0, 0, 35.4, 0.388, 50]
]  # new row of data for a new patient - prediction 0


# Predict the outcome (0 = no diabetes, 1 = diabetes)
prediction = model.predict(new_patient)
prediction2 = model.predict(outcome_zero_patient)

print(f"Diabetes Prediction (yes): {prediction[0]}")
print(f"Diabetes Prediction (no): {prediction2[0]}")
