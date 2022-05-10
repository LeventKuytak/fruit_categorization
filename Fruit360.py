import os
import tensorflow as tf
from tensorflow import keras
import numpy

zip_file = tf.keras.utils.get_file(origin="https://github.com/Horea94/Fruit-Images-Dataset.git",
                                   fname="Fruit-Images-Dataset", extract=True)
base_dir, _ = os.path.splitext(zip_file)

train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Test')

image_size = 224
batch_size = 32

train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(image_size, image_size), batch_size=batch_size)

validation_datagen = keras.preprocessing.image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(image_size, image_size), batch_size=batch_size)

IMG_SHAPE = (image_size, image_size, 3)

base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False)
base_model.trainable = False

base_model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(102, activation='sigmoid')
])

base_model.summary()