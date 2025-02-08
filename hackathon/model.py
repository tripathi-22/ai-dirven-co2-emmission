# train_model.py

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import pickle

# -----------------------------
# 1. Load and Preprocess the Data
# -----------------------------
# Load the dataset
df = pd.read_csv('co2Emissions.csv')

# Map fuel types (as in your app)
fuel_type_mapping = {
    "Z": "Premium Gasoline",
    "X": "Regular Gasoline",
    "D": "Diesel",
    "E": "Ethanol(E85)",
    "N": "Natural Gas"
}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)

# Remove rows with Natural Gas fuel type
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas", na=False)].reset_index(drop=True)

# Select the features and target variable
df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]

# Remove outliers (here using z-score threshold)
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

# -----------------------------
# 2. Define Features and Target
# -----------------------------
# Features: ensure the order is consistent with your prediction logic!
X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
y = df_new_model['CO2 Emissions(g/km)']

# -----------------------------
# 3. Train the Model
# -----------------------------
# Create and train the RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# -----------------------------
# 4. Save the Trained Model
# -----------------------------
# Write the trained model to a pickle file
with open('co2_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved as 'co2_model.pkl'")
