# --- SCRIPT 1: UPDATED MODEL WITH MobileNetV2 ---
# Includes enhanced data augmentation and a learning rate scheduler.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time

print(f"TensorFlow Version: {tf.__version__}")

# --- Paths and Constants ---
# !!! IMPORTANT !!! Update these paths to your dataset location
train_dir = '/content/dataset_split/train'
val_dir = '/content/dataset_split/val'
test_dir = '/content/dataset_split/test'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 4 # Assuming 4 classes for waste
BEST_MODEL_PATH = 'waste_classifier_mobilenet_v2_updated.keras'

# --- Data Preparation (With Enhanced Augmentation) ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2] # Added for lighting variations
)

test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

print("Setting up data generators...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- PHASE 1: INITIAL TRAINING ---
print("\nBuilding model with MobileNetV2 base...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4), # Increased dropout for better regularization
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\n🚀 Starting Phase 1: Initial Training...")
# Train the top layer
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# --- PHASE 2: FINE-TUNING ---
print("\n🚀 Starting Phase 2: Fine-Tuning...")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), # Start with a slightly higher LR for the scheduler
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Define Callbacks for Optimized Training ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=BEST_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

# --- Continue Training with Fine-Tuning and Callbacks ---
fine_tune_epochs = 25
total_epochs = 10 + fine_tune_epochs

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# --- FINAL EVALUATION ---
print("\nEvaluating the best fine-tuned model on the test set...")
# The best model is already loaded thanks to restore_best_weights=True in EarlyStopping
# and saved by ModelCheckpoint. We can directly evaluate.
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy of the Best Model: {test_acc*100:.2f}%")
print(f"✅ Final Test Loss of the Best Model: {test_loss:.4f}")

print(f"\nBest model saved to {BEST_MODEL_PATH}")