from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model('../model/model_cnn.h5')


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        processed_image = prepare_image(image, target=(256, 256))  # Adjust the target size

        prediction = model.predict(processed_image).tolist()

        return jsonify({"prediction": prediction})

    return jsonify({"error": "Invalid request method"}), 405


if __name__ == "__main__":
    app.run(debug=True)
