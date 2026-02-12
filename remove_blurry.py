import os
import cv2

dataset_dir = r"C:\Users\Lenovo\OneDrive\Desktop\AI-Anemia-Detection\dataset"
splits = ["train", "val", "test"]

THRESHOLD = 100  # blur detection threshold

def is_blurry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return True  # treat missing/corrupted images as blurry
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < THRESHOLD

for split in splits:
    split_path = os.path.join(dataset_dir, split)
    if not os.path.exists(split_path):
        print("Split not found, skipping:", split_path)
        continue

    # Loop through all class folders automatically
    for cls in os.listdir(split_path):
        cls_folder = os.path.join(split_path, cls)
        if not os.path.isdir(cls_folder):
            continue
        print("Checking", cls_folder)
        for file in os.listdir(cls_folder):
            path = os.path.join(cls_folder, file)
            if os.path.isfile(path) and is_blurry(path):
                print("Deleting blurry:", path)
                os.remove(path)

print("✅ Blurry image removal completed.")
