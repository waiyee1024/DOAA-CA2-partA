from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf

app = Flask(__name__)

# ====== 模型路径（你的 .h5 文件放这里）======
MODEL_PATHS = {
    "m1": "models/model23x23.h5",
    "m2": "models/model_101x101.h5",
}

# ====== 两个模型的输入尺寸（你按你模型训练时的输入改）======
# 例如你之前做过 23x23 和 101x101
MODEL_INPUT_SIZE = {
    "m1": 23,
    "m2": 101,
}

# ====== 可选：你的类别名字（如果没有就先用 class_0, class_1...）======
CLASS_NAMES = None
# 例如：
# CLASS_NAMES = ["cabbage", "carrot", "corn", ...]  # 你的 11 类

# ====== 启动时加载两模型（更快）======
models = {}
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    models[name] = tf.keras.models.load_model(path)
    print(f"Loaded {name} from {path}")

def preprocess_image(file_bytes: bytes, target_size: int):
    # 你的模型是 grayscale：需要 (H, W, 1)
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # 灰度
    img = img.resize((target_size, target_size))

    arr = np.array(img).astype("float32") / 255.0          # (H, W)
    arr = np.expand_dims(arr, axis=-1)                     # (H, W, 1)
    arr = np.expand_dims(arr, axis=0)                      # (1, H, W, 1)
    return arr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": list(models.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.args.get("model", "m1").lower()
    if model_name not in models:
        return jsonify({"error": f"model must be one of {list(models.keys())}"}), 400

    if "image" not in request.files:
        return jsonify({"error": "Upload an image using form-data key 'image'"}), 400

    file_bytes = request.files["image"].read()

    target_size = MODEL_INPUT_SIZE.get(model_name)
    if target_size is None:
        return jsonify({"error": f"Missing input size config for {model_name}"}), 500

    x = preprocess_image(file_bytes, target_size)

    # 推理
    y = models[model_name].predict(x)
    y = np.array(y).squeeze()  # (num_classes,)

    pred_idx = int(np.argmax(y))
    confidence = float(np.max(y))

    if CLASS_NAMES and pred_idx < len(CLASS_NAMES):
        pred_label = CLASS_NAMES[pred_idx]
    else:
        pred_label = f"class_{pred_idx}"

    return jsonify({
        "model_used": model_name,
        "input_size": target_size,
        "pred_index": pred_idx,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": y.tolist()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
