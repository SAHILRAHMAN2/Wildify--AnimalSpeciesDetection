import os
import cv2
import json
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# ── CONFIG ──────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mobilenetv2_finetuned_final.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "prediction_log.json"
)

IMG_SIZE = (224, 224)
CONFIDENCE_THRESH = 0.60


# ── LOAD MODEL ───────────────────────────────────────────────

print("Loading model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ── LOAD CLASS NAMES ─────────────────────────────────────────

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} animal classes.")


# ── FLASK APP ────────────────────────────────────────────────

app = Flask(__name__)


# ── PREDICTION LOG ───────────────────────────────────────────

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)


def log_prediction(species, confidence):

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    logs.insert(0, {
        "species": species,
        "confidence": round(confidence * 100, 1),
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d")
    })

    # Keep only the latest 50 predictions
    logs = logs[:50]

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


# ── PREDICTION FUNCTION ──────────────────────────────────────

def predict_image(frame):

    # Convert OpenCV BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image
    resized = cv2.resize(rgb, IMG_SIZE)

    # Convert to array and normalize
    arr = img_to_array(resized) / 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    # Run prediction
    preds = model.predict(arr, verbose=0)[0]

    # Get top 3 predictions
    top3_idx = preds.argsort()[-3:][::-1]

    top_species = class_names[top3_idx[0]]
    top_conf = float(preds[top3_idx[0]])

    return {
        "species": top_species,
        "confidence": round(top_conf * 100, 1),
        "top3": [
            {
                "species": class_names[i],
                "confidence": round(float(preds[i]) * 100, 1)
            }
            for i in top3_idx
        ],
        "low_conf": top_conf < CONFIDENCE_THRESH,
        "last_updated": datetime.now().strftime("%H:%M:%S")
    }


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():

    return render_template(
        "index.html",
        class_count=len(class_names)
    )


@app.route("/predict_upload", methods=["POST"])
def predict_upload():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image received"
            }), 400

        # Remove the data:image/...;base64, part
        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode Base64 image
        img_bytes = base64.b64decode(image_data)

        # Convert to OpenCV image
        np_arr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        # Run prediction
        result = predict_image(frame)

        # Save prediction
        log_prediction(
            result["species"],
            result["confidence"] / 100
        )

        return jsonify(result)

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/history")
def history():

    try:

        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

        return jsonify(logs[:20])

    except Exception as e:

        print("History error:", str(e))

        return jsonify([])


# ── RUN APP ──────────────────────────────────────────────────

if __name__ == "__main__":

    print("Starting Animal Species Detection...")

    print("Open: http://localhost:5000")

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )