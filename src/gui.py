import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image, ImageTk

# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'anemia_final.keras')

# ==============================
# Load model
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model Loaded Successfully")

# ==============================
# Upload Image Function
# ==============================
def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    # Read image
    img = cv2.imread(file_path)

    img = cv2.resize(img, (224,224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # ==============================
    # Prediction
    # ==============================
    prediction = model.predict(img, verbose=0)[0][0]

    confidence = prediction * 100

    # ==============================
    # Decision
    # ==============================
    if prediction > 0.5:
        diagnosis = "Anemia Detected"
        color = "red"
    else:
        diagnosis = "No Anemia Detected"
        color = "green"

    result_text = f"{diagnosis}\n\nConfidence: {confidence:.2f}%"

    result_label.config(text=result_text, fg=color)

    # ==============================
    # Display image
    # ==============================
    display_img = Image.open(file_path).resize((200,200))

    img_tk = ImageTk.PhotoImage(display_img)

    image_label.config(image=img_tk)
    image_label.image = img_tk


# ==============================
# Webcam Demo
# ==============================
def start_webcam():
    os.system("python realtime_predict.py")


# ==============================
# GUI Window
# ==============================
root = tk.Tk()

root.title("AI Anemia Detection System")

root.geometry("500x600")

root.configure(bg="#eaf6f6")


# ==============================
# Title
# ==============================
tk.Label(
    root,
    text="AI-Based Early Anemia Detection",
    font=("Arial",18,"bold"),
    bg="#eaf6f6",
    fg="#2c7fb8"
).pack(pady=15)


# ==============================
# Upload Button
# ==============================
tk.Button(
    root,
    text="Upload Blood Image",
    command=upload_image,
    width=30,
    bg="#2c7fb8",
    fg="white",
    font=("Arial",11,"bold")
).pack(pady=10)


# ==============================
# Webcam Button
# ==============================
tk.Button(
    root,
    text="Start Webcam (Demo)",
    command=start_webcam,
    width=30,
    bg="#16a085",
    fg="white",
    font=("Arial",11,"bold")
).pack(pady=10)


# ==============================
# Image Preview
# ==============================
image_label = tk.Label(root,bg="#eaf6f6")

image_label.pack(pady=15)


# ==============================
# Result Display
# ==============================
result_label = tk.Label(
    root,
    font=("Arial",14,"bold"),
    justify="center",
    bg="white",
    width=35,
    height=6,
    relief="solid",
    bd=1
)

result_label.pack(pady=15)


# ==============================
# Footer
# ==============================
tk.Label(
    root,
    text="AI Healthcare Diagnostic Tool",
    font=("Arial",9),
    bg="#eaf6f6",
    fg="gray"
).pack(side="bottom", pady=10)


# ==============================
# Run App
# ==============================
root.mainloop()