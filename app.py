from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RF_MODEL_DIR = Path("random_forest_station_models")


def load_rf_model(station):
    station_clean = station.replace(" ", "_").replace("/", "_")
    model_path = RF_MODEL_DIR / f"rf_model_{station_clean}.joblib"

    if not model_path.exists():
        return None
    return joblib.load(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    model_type = data.get("model_type")
    station = data.get("station")
    year = data.get("year")
    month = data.get("month")

    # Validate input
    if not model_type or not station or year is None or month is None:
        return jsonify({"error": "Missing required parameters"}), 400

    if model_type != "random_forest":
        return jsonify({"error": "Only random_forest model_type is supported now"}), 400


    model = load_rf_model(station)
    if model is None:
        return jsonify({"error": f"No RF model found for station {station}"}), 404

    X = np.array([[year, month]])

    preds = model.predict(X)
    day_pred = float(preds[0][0])
    night_pred = float(preds[0][1])

    return jsonify({
        "station": station,
        "model_type": "random_forest",
        "day": day_pred,
        "night": night_pred
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
