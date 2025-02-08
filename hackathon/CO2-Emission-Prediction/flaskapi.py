# flask_api.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model from a pickle file.
# (Make sure you have run a training script that saves the model as 'co2_model.pkl'.)
with open("co2_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting a JSON payload with a key "features".
        data = request.get_json()
        # Convert the incoming features into a NumPy array. Ensure the order matches your training!
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"CO2_Emissions": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/global_avg", methods=["GET"])
def global_avg():
    try:
        # For demonstration purposes – replace with your real logic/API call if needed.
        global_emission_avg = 180  # example value
        return jsonify({"Global_Avg_CO2": global_emission_avg})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Running on port 5000 by default.
    app.run(debug=True, host='0.0.0.0', port=5000)
