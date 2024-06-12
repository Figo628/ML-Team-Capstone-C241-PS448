import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

val_dir = '../lettuce_experimental/valid'
model = tf.keras.models.load_model('../model/model_cnn.h5')

# Membuat ImageDataGenerator untuk data validasi
datagen = ImageDataGenerator(rescale=1./255)

# Memuat data validasi menggunakan generator
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
)

# Evaluasi model pada data uji menggunakan generator
loss, accuracy = model.evaluate(val_generator)
print("Test Accuracy:", accuracy)

# Prediksi menggunakan model pada data uji
y_pred = model.predict(val_generator)

# Ambil label sebenarnya dari generator data
y_true = val_generator.classes

# Untuk model klasifikasi multiclass, ambil label dengan nilai terbesar sebagai prediksi
y_pred_classes = tf.argmax(y_pred, axis=1)

# Metrik presisi
precision = precision_score(y_true, y_pred_classes, average='weighted')
print("Precision:", precision)

# Metrik recall
recall = recall_score(y_true, y_pred_classes, average='weighted')
print("Recall:", recall)

# Metrik F1-score
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print("F1-score:", f1)