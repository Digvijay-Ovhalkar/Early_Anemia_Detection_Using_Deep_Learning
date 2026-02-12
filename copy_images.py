import os
import shutil

# Define source folders (where your images currently are)
source_folders = [
    r"C:\Users\Lenovo\Downloads\dataset_images\dataset2-master",
    r"C:\Users\Lenovo\Downloads\dataset_images\dataset-master\dataset-master"
]

# Define destination folders (your project dataset folders)
dest_folders = {
    "train": r"C:\Users\Lenovo\OneDrive\Desktop\AI-Anemia-Detection\dataset\train",
    "val": r"C:\Users\Lenovo\OneDrive\Desktop\AI-Anemia-Detection\dataset\val",
    "test": r"C:\Users\Lenovo\OneDrive\Desktop\AI-Anemia-Detection\dataset\test"
}

# You can split images randomly between train, val, test if needed
import random

# List all image extensions you want to copy
image_extensions = [".jpg", ".jpeg", ".png"]

# Function to get all images from a folder
def get_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                images.append(os.path.join(root, file))
    return images

# Copy images
for src_folder in source_folders:
    images = get_images(src_folder)
    random.shuffle(images)  # shuffle for random distribution
    
    total = len(images)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    for i, img_path in enumerate(images):
        if i < train_split:
            dest = dest_folders["train"]
        elif i < val_split:
            dest = dest_folders["val"]
        else:
            dest = dest_folders["test"]

        shutil.copy(img_path, dest)
        print(f"Copied {img_path} -> {dest}")

print("All images copied successfully!")
