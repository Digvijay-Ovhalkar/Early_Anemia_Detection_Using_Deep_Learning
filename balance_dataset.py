import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================================
# CONFIG
# ================================
dataset_dir = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/train"
target_count = 800   # Change to 1000 if you want more

# ================================
# AUGMENTATION SETUP
# ================================
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# ================================
# BALANCING PROCESS
# ================================
for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    print(f"\n📁 {cls}: {len(images)} images")

    if len(images) >= target_count:
        print("   ✅ Already sufficient, skipping")
        continue

    while len(images) < target_count:
        img_name = random.choice(images)
        img_path = os.path.join(cls_path, img_name)

        try:
            # Load image safely and convert to RGB
            img = Image.open(img_path).convert("RGB")
            img_arr = np.expand_dims(np.array(img), 0)

            # Generate augmented image
            aug_iter = datagen.flow(img_arr, batch_size=1)
            aug_img = next(aug_iter)[0].astype("uint8")

            # Convert to PIL image
            img_out = Image.fromarray(aug_img)

            if img_out.mode != "RGB":
                img_out = img_out.convert("RGB")

            new_name = f"aug_{len(images)}.jpg"
            save_path = os.path.join(cls_path, new_name)

            img_out.save(save_path, "JPEG")

            images.append(new_name)

        except Exception as e:
            print(f"⚠ Skipping {img_name}: {e}")

    print(f"   🎯 Balanced to {len(images)} images")

print("\n✅ Dataset balancing completed successfully!")
