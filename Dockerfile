FROM tensorflow/serving

# Copy the model to the model folder inside the container
COPY ./model/model_cnn_same_like_before /models/model_cnn_same_like_before

# Set environment variable to specify model name
ENV MODEL_NAME=model_cnn_same_like_before

# Expose ports (optional, default ports are used if not specified)
EXPOSE 8500
EXPOSE 8501