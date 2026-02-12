import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image, ImageTk

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'anemia_final.keras')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train')

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load correct class names
CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

print("Loaded classes:", CLASS_NAMES)

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))

    # SAME preprocessing as training (MobileNetV2)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]

    # 🔥 Show ALL class predictions
    result_text = "All Class Predictions:\n\n"
    for i, class_name in enumerate(CLASS_NAMES):
        result_text += f"{class_name:15s} : {predictions[i]*100:.2f}%\n"

    result_label.config(text=result_text)

    display_img = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(display_img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

def start_webcam():
    os.system("python realtime_predict.py")

root = tk.Tk()
root.title("AI Anemia Detection System")
root.geometry("460x620")

tk.Label(
    root,
    text="AI Anemia Detection System",
    font=("Arial", 16, "bold")
).pack(pady=10)

tk.Button(
    root,
    text="Upload Blood Cell Image",
    command=upload_image,
    width=30
).pack(pady=10)

tk.Button(
    root,
    text="Start Webcam (Demo)",
    command=start_webcam,
    width=30
).pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    font=("Courier", 10),
    justify="left"
)
result_label.pack(pady=10)

root.mainloop()
