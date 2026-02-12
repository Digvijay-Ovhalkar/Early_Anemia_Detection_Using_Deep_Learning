import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import os

# =====================================================
# PATHS
# =====================================================
test_path = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/test"
model_path = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/models/anemia_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =====================================================
# LOAD TEST DATASET
# =====================================================
test_ds = image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize (same as training)
normalization = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# =====================================================
# EVALUATE
# =====================================================
loss, accuracy = model.evaluate(test_ds)
print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
