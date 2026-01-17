from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from openai import OpenAI
import os
import json

app = Flask(__name__)
CORS(app)


# ===== Модель =====
model = tf.keras.models.load_model("waste_model.h5")

CLASSES = [
    "Пластик",
    "Бумага",
    "Стекло",
    "Металл",
    "Органика",
    "Опасные отходы"
]

UTILIZATION = {
    "Стекло": "Сдать в контейнер для стекла или пункт приёма стеклотары",
    "Пластик": "Промыть и сдать в контейнер для пластика",
    "Бумага": "Сдать в контейнер для макулатуры",
    "Металл": "Сдать в контейнер для металла",
    "Органика": "Компостирование",
    "Опасные отходы": "Специализированный пункт (батарейки, лампы)"
}

@app.route("/")
def home():
    return render_template("index.html")

# ===== Подготовка изображения =====
def prepare_image(image: Image.Image):
    _, h, w, _ = model.input_shape  # берём размер из модели
    image = image.resize((w, h))
    img = np.array(image).astype("float32") / 255.0
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    return np.expand_dims(img, axis=0)

@app.route("/scan")
def scan():
    return render_template("scan.html")

# ===== Распознавание изображения =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Файл не получен"}), 400

        image = Image.open(request.files["image"]).convert("RGB")
        img = prepare_image(image)

        preds = model.predict(img)
        idx = int(np.argmax(preds))
        waste_type = CLASSES[idx]

        return jsonify({
            "type": waste_type,
            "instruction": UTILIZATION[waste_type]
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
