import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import os

# =====================================================
# GPU MEMORY SETUP
# =====================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

tf.keras.backend.clear_session()

# =====================================================
# PATHS
# =====================================================
train_path = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/train"
val_path   = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/val"
test_path  = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_INITIAL = 30
EPOCHS_FINE = 10

# =====================================================
# DATASET CHECK
# =====================================================
def check_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        raise ValueError(f"No class subfolders found in: {path}")
    print(f"✅ Dataset found at: {path}")
    print(f"Classes: {subdirs}")

check_dataset(train_path)
check_dataset(val_path)
check_dataset(test_path)

# =====================================================
# LOAD DATASETS
# =====================================================
train_ds = image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("✅ Classes used:", class_names)

# =====================================================
# PERFORMANCE OPTIMIZATION
# =====================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# =====================================================
# NORMALIZATION + AUGMENTATION
# =====================================================
normalization = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# =====================================================
# TRANSFER LEARNING MODEL
# =====================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

for layer in base_model.layers[:-80]:
    layer.trainable = False


inputs = layers.Input(shape=(224,224,3))
x = normalization(inputs)
x = data_augmentation(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)

if num_classes == 2:
    outputs = layers.Dense(1, activation='sigmoid')(x)
    loss_fn = 'binary_crossentropy'
else:
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    loss_fn = 'sparse_categorical_crossentropy'
    
# Print AFTER assignment
print("Loss:", loss_fn)   # <-- Step 4: Verify loss function    

model = models.Model(inputs, outputs)
model.summary()

# =====================================================
# COMPILE INITIAL MODEL
# =====================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

# =====================================================
# CALLBACKS
# =====================================================
checkpoint_path = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/models/anemia_best.keras"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    patience=6,
    restore_best_weights=True
)

# =====================================================
# OPTIONAL CLASS WEIGHTS (EDIT IF IMBALANCED)
# =====================================================
class_weight = None


# =====================================================
# TRAIN STAGE 1 (FEATURE EXTRACTION)
# =====================================================
print("\n🚀 Starting Initial Training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    callbacks=[checkpoint_cb, early_stop_cb],
    class_weight=class_weight
)

# =====================================================
# FINE-TUNING
# =====================================================
print("\n🔧 Starting Fine-Tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint_cb, early_stop_cb],
    class_weight=class_weight
)

# =====================================================
# SAVE FINAL MODEL
# =====================================================
final_model_path = r"C:/Users/Lenovo/OneDrive/Desktop/AI-Anemia-Detection/src/models/anemia_final.keras"
model.save(final_model_path)
print(f"✅ Final model saved at: {final_model_path}")

# =====================================================
# TEST EVALUATION
# =====================================================
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

print("\n✅ Training Complete!")
