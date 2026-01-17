import tensorflow as tf

model = tf.keras.models.load_model("waste_model.h5")
model.summary()