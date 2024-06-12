docker pull tensorflow/serving

DIR_MODEL="C:/Users/IDe/PycharmProjects/ML-Team-Capstone-C241-PS448/model/model_cnn_same_like_before"

docker run -t --rm -p 8501:8501 \
  -v "$DIR_MODEL:/models/model_cnn_same_like_before" \
  -e MODEL_NAME=model_cnn_same_like_before \
  tensorflow/serving &