import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# PATHS
# ==============================

DATASET_PATH = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/train"
MODEL_SAVE_PATH = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/models/anemia_final.keras"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

# ==============================
# DATA AUGMENTATION
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.15,
    brightness_range=[0.8,1.2]
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)

# ==============================
# CLASS WEIGHTS (BALANCE DATA)
# ==============================

y_train = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ==============================
# TRANSFER LEARNING MODEL
# ==============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# ==============================
# TRAIN MODEL
# ==============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ==============================
# FINE TUNING
# ==============================

print("\nStarting Fine-Tuning...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# ==============================
# SAVE MODEL
# ==============================

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model.save(MODEL_SAVE_PATH)

print("Model saved at:", MODEL_SAVE_PATH)

# ==============================
# PLOT ACCURACY GRAPH
# ==============================

plt.figure()
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_graph.png")
plt.show()

# ==============================
# PLOT LOSS GRAPH
# ==============================

plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_graph.png")
plt.show()

# ==============================
# CONFUSION MATRIX
# ==============================

val_generator.reset()

pred = model.predict(val_generator)

pred_classes = np.argmax(pred, axis=1)
true_classes = val_generator.classes

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            fmt="d")

plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.show()

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report\n")

print(classification_report(
    true_classes,
    pred_classes,
    target_names=CLASS_NAMES
))