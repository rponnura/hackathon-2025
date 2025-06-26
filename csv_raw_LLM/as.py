from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# === Load model and encoders ===
model = joblib.load("model.pkl")
target_encoders = joblib.load("encoders.pkl")
output_cols = list(target_encoders.keys())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expect a dict with Feature1, Feature2, etc.
    print(data)
    input_df = pd.DataFrame([data])

    try:
        preds = model.predict(input_df)[0]
        decoded = {
            col: target_encoders[col].inverse_transform([preds[i]])[0]
            for i, col in enumerate(output_cols)
        }
        return jsonify(decoded)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
