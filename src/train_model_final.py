import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Paths and Constants ---
train_dir = '/content/dataset_split/train'
val_dir = '/content/dataset_split/val'
test_dir = '/content/dataset_split/test'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 4
BEST_MODEL_PATH = 'waste_classifier_final.keras' # Using the recommended .keras format

# --- Data Preparation ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

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
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Starting Phase 1: Initial Training...")
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use a very low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Define Callbacks for Optimization ---
# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=4, # Wait 4 epochs before stopping
    restore_best_weights=True # Automatically restore the weights from the best epoch
)
# Save only the best model based on validation accuracy
model_checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# --- Continue Training with Fine-Tuning and Callbacks ---
fine_tune_epochs = 20 # Set a max number of epochs
total_epochs = 10 + fine_tune_epochs

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint] # Add the callbacks here
)

# --- FINAL EVALUATION ---
# The best model is already saved by ModelCheckpoint, and EarlyStopping restored the best weights
print("\nEvaluating the best fine-tuned model on the test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy of the Best Model: {test_acc*100:.2f}%")

print(f"\nBest model saved to {BEST_MODEL_PATH}")