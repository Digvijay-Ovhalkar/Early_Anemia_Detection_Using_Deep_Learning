import os
from PIL import Image

dataset_dir = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset"

def clean_folder(folder):
    removed = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                os.remove(path)
                removed += 1
    return removed

removed = clean_folder(dataset_dir)
print(f"✅ Removed corrupted images: {removed}")
