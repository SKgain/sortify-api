import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam # Import Adam to customize it

# Paths
train_dir = 'dataset_split/train'
val_dir = 'dataset_split/val'
test_dir = 'dataset_split/test'

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

# Data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
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


model = models.Sequential([
    # Block 1
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)), # CHANGED: Increased filters
    layers.BatchNormalization(), # ADDED: Batch Normalization
    layers.MaxPooling2D(2,2),
    
    # Block 2
    layers.Conv2D(128, (3,3), activation='relu'), # CHANGED: Increased filters
    layers.BatchNormalization(), # ADDED: Batch Normalization
    layers.MaxPooling2D(2,2),
    
    # Block 3
    layers.Conv2D(256, (3,3), activation='relu'), # CHANGED: Increased filters
    layers.BatchNormalization(), # ADDED: Batch Normalization
    layers.MaxPooling2D(2,2),

    # Block 4 - The new block
    layers.Conv2D(512, (3,3), activation='relu'), # ADDED: New Layer
    layers.BatchNormalization(), # ADDED: Batch Normalization
    layers.MaxPooling2D(2,2), # ADDED: New Layer

    # Classifier Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- Adjusting the Training Process ---

# CHANGED: Using a custom Adam optimizer with a smaller learning rate
optimizer = Adam(learning_rate=0.0001) 

# Compile model with the new optimizer
model.compile(
    optimizer=optimizer, # CHANGED
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=25, # CHANGED: Increased epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Save the new tuned model with a different name
model.save('waste_classifier_tuned_cnn.h5') # CHANGED: New model name
print("Model saved as waste_classifier_tuned_cnn.h5")