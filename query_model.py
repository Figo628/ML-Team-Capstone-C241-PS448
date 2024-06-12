# import tensorflow as tf
# import numpy as np
# import sys
# from PIL import Image
# import requests
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# def load_image(image_path, target_size):
#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
#
# def predict(image_array):
#     server_url = 'http://localhost:8501/v1/models/model_cnn_same_like_before:predict'
#     headers = {"content-type": "application/json"}
#     data = {"instances": image_array.tolist()}
#     response = requests.post(server_url, headers=headers, json=data)
#     prediction_result = response.json()
#     return prediction_result
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python query_model.py <image_path>")
#         sys.exit(1)
#
#     image_path = sys.argv[1]
#     target_size = (300, 300)
#
#     datagen = ImageDataGenerator(rescale=1./255.)
#     image_array = load_image(image_path, target_size)
#     image_array = datagen.flow(image_array, batch_size=1)[0]
#
#     prediction = predict(image_array)
#     print(prediction)
import base64

# import tensorflow as tf
# import numpy as np
# import sys
# from PIL import Image
# import requests
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# def load_image(image_path, target_size):
#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
#
# def predict(image_array):
#     server_url = 'http://localhost:8501/v1/models/model_cnn_same_like_before:predict'
#     headers = {"content-type": "application/json"}
#     data = {"instances": image_array.tolist()}
#     response = requests.post(server_url, headers=headers, data=data)
#     if response.status_code == 200:
#         prediction_result = json.loads(response.text)['predictions']
#         return prediction_result
#     else:
#         print("Failed to receive valid response from server.")
#         return None
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python query_model.py <image_path>")
#         sys.exit(1)
#
#     image_path = sys.argv[1]
#     target_size = (300, 300)
#
#     datagen = ImageDataGenerator(rescale=1./255.)
#     image_array = load_image(image_path, target_size)
#     image_array = datagen.flow(image_array, batch_size=1)[0]
#
#     prediction = predict(image_array)
#     if prediction is not None:
#         print("Prediction result:")
#         print(prediction)
#     else:
#         print("Failed to make prediction.")

import numpy as np
import io
from PIL import Image
import requests
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def load_and_preprocess_image(path_dir, target_size):
    img = load_img(path_dir, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    gene_image = ImageDataGenerator(
        rescale=1. / 255,
    )
    gene = gene_image.flow(
        img_array,
        batch_size=1,
    )
    image = next(gene)
    print(f'Image shape: {image.shape}')

    return image


def predict(temp_img):
    server_url = 'http://localhost:8501/v1/models/model_cnn_same_like_before:predict'
    headers = {"Content-Type": "application/json"}
    data = json.dumps({'instances': temp_img.tolist()})

    response = requests.post(server_url, headers=headers, data=data)

    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        print("Error:", response.text)
        return None


if __name__ == '__main__':
    # Ganti dengan path gambar Anda
    image_path = './gambar_predict/lettuce_.jpg'
    target_size = (256, 256)  # Ganti dengan ukuran input yang diharapkan oleh model Anda

    image_array = load_and_preprocess_image(image_path, target_size)
    prediction = predict(image_array)
    print(prediction)
