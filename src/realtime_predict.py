import os
import cv2
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'anemia_final.keras')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train')

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print("Loaded classes:", CLASS_NAMES)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]
    top_index = np.argmax(predictions)

    label = CLASS_NAMES[top_index]
    confidence = predictions[top_index] * 100

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Anemia Detection (Webcam Demo)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
