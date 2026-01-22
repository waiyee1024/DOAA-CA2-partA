from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import os
import time
import traceback
import tensorflow as tf

# -----------------------------------
# Config
# -----------------------------------
app = Flask(__name__)

MODEL_PATHS = {
    "m1": "models/model_23x23",
    "m2": "models/model_101x101",
}

MODEL_INPUT_SIZE = {"m1": 23, "m2": 101}

MODEL_INFO = {
    "m1": {"description": "Lightweight model (23x23) optimized for faster inference"},
    "m2": {"description": "Higher-resolution model (101x101) optimized for higher confidence predictions"},
}

CLASS_NAMES = [
    "Bean",
    "Bitter_Gourd",
    "Brinjal",
    "Cabbage",
    "Capsicum",
    "Cauliflower and Broccoli",
    "Cucumber and Bottle_Gourd",
    "Potato",
    "Pumpkin",
    "Radish and Carrot",
    "Tomato",
]


# -----------------------------------
# Helpers
# -----------------------------------
def preprocess_image(file_bytes: bytes, target_size: int) -> np.ndarray:
    """
    Convert incoming image bytes to grayscale (L), resize to model input,
    normalize to [0,1], and reshape to (1, H, W, 1).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((target_size, target_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
    arr = np.expand_dims(arr, axis=0)   # (1, H, W, 1)
    return arr


def index_to_label(idx: int) -> str:
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"class_{idx}"


def detect_signature_endpoint(savedmodel_dir: str) -> str:
    """
    Auto-detect available signature endpoints in SavedModel.
    Prefer 'serving_default', then 'serve', else pick the first available.
    """
    obj = tf.saved_model.load(savedmodel_dir)
    keys = list(obj.signatures.keys())
    if not keys:
        raise RuntimeError(f"No signatures found in SavedModel: {savedmodel_dir}")

    if "serving_default" in keys:
        return "serving_default"
    if "serve" in keys:
        return "serve"
    return keys[0]


def load_savedmodel_layer(savedmodel_dir: str) -> tf.keras.layers.Layer:
    endpoint = detect_signature_endpoint(savedmodel_dir)
    print(f"[INFO] Using signature endpoint '{endpoint}' for {savedmodel_dir}")
    return tf.keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=endpoint)


def extract_output_to_numpy(model_output) -> np.ndarray:
    """
    Extract model output to 1D numpy array (num_classes,).
    Handles dict/list/tensor outputs and applies softmax if needed.
    """
    if isinstance(model_output, dict):
        first_key = next(iter(model_output.keys()))
        y = model_output[first_key]
    elif isinstance(model_output, (list, tuple)):
        y = model_output[0]
    else:
        y = model_output

    y = tf.convert_to_tensor(y)
    y = tf.squeeze(y, axis=0)  # (num_classes,)
    y_np = y.numpy()

    # If output is logits (not probabilities), convert to probabilities.
    s = float(np.sum(y_np))
    looks_like_prob = (0.98 <= s <= 1.02) and np.all(y_np >= 0) and np.all(y_np <= 1)
    if not looks_like_prob:
        y_np = tf.nn.softmax(y, axis=-1).numpy()

    return y_np


# -----------------------------------
# Load models on startup
# -----------------------------------
models = {}
for model_key, model_dir in MODEL_PATHS.items():
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    models[model_key] = load_savedmodel_layer(model_dir)
    print(f"[OK] Loaded {model_key} from {model_dir}")


# -----------------------------------
# Routes
# -----------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "model_details": {
            k: {
                "input_size": MODEL_INPUT_SIZE.get(k),
                "description": MODEL_INFO[k]["description"]
            } for k in models
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Teacher requirement: only show predicted class name in response.
    """
    try:
        model_name = request.args.get("model", "m1").lower()
        if model_name not in models:
            return jsonify({"error": f"model must be one of {list(models.keys())}"}), 400

        if "image" not in request.files:
            return jsonify({"error": "Upload an image using form-data key 'image'"}), 400

        target_size = MODEL_INPUT_SIZE.get(model_name)
        if target_size is None:
            return jsonify({"error": f"Missing input size config for {model_name}"}), 500

        file_bytes = request.files["image"].read()
        x_np = preprocess_image(file_bytes, target_size)

        # Inference
        _ = time.time()
        out = models[model_name](tf.convert_to_tensor(x_np, dtype=tf.float32))
        y = extract_output_to_numpy(out)

        pred_idx = int(np.argmax(y))
        pred_label = index_to_label(pred_idx)

        # ✅ Only return predicted class name (as requested by teacher)
        return jsonify({
            "predicted_class_name": pred_label
        })

    except Exception as e:
        # Helpful error details for local debugging
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
