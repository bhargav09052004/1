from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Load and preprocess data
data = pd.read_csv('weather_data.csv')
data['HeatWave'] = data['Temperature'] > 30  # Example condition for heat wave

# Features and target
X = data[['Temperature', 'Humidity']]
y = data['HeatWave']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'heat_wave_model.pkl')
print("Model saved as heat_wave_model.pkl")

# Load the model
loaded_model = joblib.load('heat_wave_model.pkl')
print("Model loaded from heat_wave_model.pkl")

# Example usage
example_data = [[35, 45]]  # Example input: Temperature=35, Humidity=45
prediction = loaded_model.predict(example_data)
print(f"Heat wave prediction for {example_data}: {prediction[0]}")
