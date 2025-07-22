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
BEST_MODEL_PATH = 'waste_classifier_best_model.h5' # File to save the best model

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

# --- High-Accuracy Model: Transfer Learning with MobileNetV2 ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze the base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- Compile and Define Callbacks ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping to prevent overfitting and save time
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# ModelCheckpoint to save only the best model
model_checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# --- Train the Model ---
print("🚀 Starting training with Transfer Learning...")
history = model.fit(
    train_generator,
    epochs=20, # Set a higher max epoch; EarlyStopping will stop it sooner
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint] # Add the callbacks
)

# --- Evaluate and Save ---
print("\nLoading the best model for final evaluation...")
# Load the best model saved by ModelCheckpoint
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

print("\nEvaluating best model on the test set...")
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy of the Best Model: {test_acc*100:.2f}%")