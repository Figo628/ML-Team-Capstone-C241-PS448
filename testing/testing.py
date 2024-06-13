# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
# from PIL import Image
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score

# image_dir = '../gambar_predict/lettuce_.jpg'
# model = tf.keras.models.load_model('../model/model_cnn.h5')
#
# # Membuat ImageDataGenerator untuk data validasi
# datagen = ImageDataGenerator(rescale=1./255)
#
# # Memuat data validasi menggunakan generator
# val_generator = datagen.flow_from_directory(
#     val_dir,
#     target_size=(256, 256),
# )

# # Evaluasi model pada data uji menggunakan generator
# loss, accuracy = model.evaluate(val_generator)
# print("Test Accuracy:", accuracy)
#
# # Prediksi menggunakan model pada data uji
# y_pred = model.predict(val_generator)
#
# # Ambil label sebenarnya dari generator data
# y_true = val_generator.classes
#
# # Untuk model klasifikasi multiclass, ambil label dengan nilai terbesar sebagai prediksi
# y_pred_classes = tf.argmax(y_pred, axis=1)
#
# # Metrik presisi
# precision = precision_score(y_true, y_pred_classes, average='weighted')
# print("Precision:", precision)
#
# # Metrik recall
# recall = recall_score(y_true, y_pred_classes, average='weighted')
# print("Recall:", recall)
#
# # Metrik F1-score
# f1 = f1_score(y_true, y_pred_classes, average='weighted')
# print("F1-score:", f1)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report, auc
from sklearn.preprocessing import label_binarize

# Path to the test directory
test_data_dir = '../gambar_predict/'

# ImageDataGenerator for preprocessing images
test_data_generator = ImageDataGenerator(rescale=1./255)

# Flow images from directory
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(256, 256),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# Load the model
model = tf.keras.models.load_model('../model/model_cnn.h5')

# Predict probabilities
probabilities = model.predict(test_generator)

# Assume you have the true labels for the test data
# Example: test_labels = [label1, label2, label3, ...]
# Replace it with your actual test labels
# test_labels = [...]

# For demonstration purposes, let's create dummy test labels
# You should replace these with your actual test labels
test_labels = test_generator.classes

# Binarize the labels
test_labels_binary = label_binarize(test_labels, classes=[0, 1, 2, 3])

# Function to evaluate metrics at a given threshold
def evaluate_threshold(probabilities, test_labels, threshold):
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='micro')  # Adjust the average parameter as needed
    recall = recall_score(test_labels, predictions, average='micro')  # Adjust the average parameter as needed
    f1 = f1_score(test_labels, predictions, average='micro')  # Adjust the average parameter as needed
    return accuracy, precision, recall, f1

# Evaluate metrics at various thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
results = {threshold: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for threshold in thresholds}

for threshold in thresholds:
    accuracy, precision, recall, f1 = evaluate_threshold(probabilities, test_labels_binary, threshold)
    results[threshold]['accuracy'].append(accuracy)
    results[threshold]['precision'].append(precision)
    results[threshold]['recall'].append(recall)
    results[threshold]['f1'].append(f1)

# Display the results
for threshold, metrics in results.items():
    print(f'Threshold: {threshold:.1f}')
    print(f'Accuracy: {np.mean(metrics["accuracy"]):.4f}, Precision: {np.mean(metrics["precision"]):.4f}, '
          f'Recall: {np.mean(metrics["recall"]):.4f}, F1 Score: {np.mean(metrics["f1"]):.4f}\n')

# ROC curve
plt.figure()
for i in range(4):  # Number of classes
    fpr, tpr, _ = roc_curve(test_labels_binary[:, i], probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

