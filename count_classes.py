import os

dataset_dir = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/train"

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"{cls}: {count}")
