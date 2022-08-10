from lib2to3.pytree import convert
from keras.models import load_model
import tensorflow as tf

model = load_model("HDF.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Model Converted Successfully !")


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

