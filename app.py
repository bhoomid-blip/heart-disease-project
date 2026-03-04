from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# ======================
# Load trained model
# ======================
model = pickle.load(open("heart_model.pkl", "rb"))

# ======================
# Home route
# ======================
@app.route("/")
def home():
    return render_template("index.html")

# ======================
# Prediction route
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age = float(data["age"])
        sex = float(data["sex"])
        bp = float(data["bp"])
        chol = float(data["cholestrol"])

        # IMPORTANT: same order as training
        features = np.array([[age, sex, bp, chol]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": "Error in prediction"})

# ======================
# Run app
# ======================
if __name__ == "__main__":
    app.run(debug=True)