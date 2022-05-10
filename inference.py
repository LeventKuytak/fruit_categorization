import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub

im_dir="./Images"

#python dosyalarının olduğu dizine resimler yüklenir
base_dir = os.path.join(os.path.dirname(im_dir), 'Images')
test_dir = os.path.join(base_dir, 'Test')

BATCH_SIZE = 256
IMG_SHAPE  = 224

image_gen_test = ImageDataGenerator(rescale=1./255)

#Test image Generator
test_generator = image_gen_test.flow_from_directory(directory=test_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode="categorical")

#model load
saved_keras_model_filepath = './model.h5'

reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()

#Test image oluşturma
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
test=test_generator
pred=reloaded_keras_model.predict(test,
steps=STEP_SIZE_TEST,
verbose=1)

#Prediction label yaratma
predicted_class_indices=np.argmax(pred,axis=1)
labels = (test.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#Örnek 30 resim ve prediction
for n in range(30):
    plt.imshow(test[1][0][n])
    plt.title(predictions[n+256])
    plt.show()  

#Test image için Loss ve Accuracy
loss, accuracy = reloaded_keras_model.evaluate(test,steps=STEP_SIZE_TEST,
verbose=1)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

