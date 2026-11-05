import tensorflow as tf

modelo = tf.keras.models.load_model('modelo_basura_alta_precision.keras')

conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
conv.optimizations = [tf.lite.Optimize.DEFAULT]   # cuantización dinámica

tflite = conv.convert()

with open('modelo.tflite', 'wb') as f:
    f.write(tflite)

print("Listo. Tamaño:", round(len(tflite) / 1024 / 1024, 2), "MB")
