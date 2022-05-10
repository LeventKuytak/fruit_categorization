import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub

im_dir="./Images"

#python dosyalarının olduğu dizine resimler yüklenir
#Resimlerin %60'ı train, %20'si validation, %20'si test datası olacak şekilde ayrılır
base_dir = os.path.join(os.path.dirname(im_dir), 'Images')
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

BATCH_SIZE = 256
IMG_SHAPE  = 224


#Trian image Generator
image_gen_train = ImageDataGenerator(rescale=1./255)
                                     #rotation_range=40,
                                     #width_shift_range=0.2,
                                     #height_shift_range=0.2,
                                     #shear_range=0.2,
                                     #zoom_range=0.2,
                                     #horizontal_flip=True,
                                     #fill_mode='nearest')


train_generator = image_gen_train.flow_from_directory(directory=train_dir,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          target_size=(IMG_SHAPE,IMG_SHAPE),
                                          class_mode="categorical")

#Validation image Generator
image_gen_val = ImageDataGenerator(rescale=1./255)

valid_generator = image_gen_val.flow_from_directory(directory=validation_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode="categorical")

#Test image Generator
image_gen_test = ImageDataGenerator(rescale=1./255)

test_generator = image_gen_test.flow_from_directory(directory=test_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode="categorical")

#Mobile_Net Transfer Learning Algoritması
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224,3))

feature_extractor.trainable = False

#Arka Planı Temizleyelim
tf.keras.backend.clear_session()

#Transfer Learning ile Sequential Model Oluşturma
model = tf.keras.Sequential([
        feature_extractor])

layer_neurons = [512, 256, 128, 56]

#Modele Dense ve Dropout Layer ekleme
for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
            
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.summary()

#Model compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Early Stop
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#Model Fit      
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,
                    callbacks=[early_stopping])

#Test image oluşturma
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
test=test_generator
pred=model.predict(test,
steps=STEP_SIZE_TEST,
verbose=1)

#Prediction label yaratma
predicted_class_indices=np.argmax(pred,axis=1)
labels = (test.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#Örnek 30 resim ve prediction
for n in range(30):
    plt.imshow(test[0][0][n])
    plt.title(predictions[n])
    plt.show()  

#Test image için Loss ve Accuracy
loss, accuracy = model.evaluate(test,steps=STEP_SIZE_TEST,
verbose=1)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))


#model save
saved_keras_model_filepath = './model.h5'

model.save(saved_keras_model_filepath)