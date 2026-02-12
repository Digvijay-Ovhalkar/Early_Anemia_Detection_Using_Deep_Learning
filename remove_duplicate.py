import os
from PIL import Image
import imagehash

dataset_dir = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset"

hashes = {}

removed = 0
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            h = imagehash.phash(img)
            if h in hashes:
                os.remove(path)
                removed += 1
            else:
                hashes[h] = path
        except:
            pass

print(f"✅ Removed duplicate images: {removed}")
