import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)
