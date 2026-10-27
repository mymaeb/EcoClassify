import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import os

# 1. Configuración de hiperparámetros
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 25

# 2. Datos
print("Cargando datasets...")
train_ds = image_dataset_from_directory(
    'dataset/train', 
    image_size=IMG_SIZE,
    batch_size=BATCH, 
    label_mode='categorical', 
    shuffle=True
)

val_ds = image_dataset_from_directory(
    'dataset/validation', 
    image_size=IMG_SIZE,
    batch_size=BATCH, 
    label_mode='categorical', 
    shuffle=False
)

CLASSES = train_ds.class_names
NUM_CLASSES = len(CLASSES)
print(f"Clases detectadas ({NUM_CLASSES}):", CLASSES)

with open('classes.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(CLASSES))

# Capas de aumento de datos
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

# Optimizar rendimiento con prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 3. Modelo base (EfficientNetB0)
print("Construyendo el modelo...")
base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)

# Fine-tuning: entrenar las últimas 40 capas
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    data_augmentation,
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# 4. Compilación
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=2, 
        min_lr=1e-6,
        verbose=1
    )
]

# 6. Entrenamiento
print("Iniciando entrenamiento...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=callbacks
)

# 7. Guardar modelo
model.save('modelo_basura_alta_precision.keras')
print("¡Modelo guardado exitosamente como 'modelo_basura_alta_precision.keras'!")
