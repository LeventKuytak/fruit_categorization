import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

im_dir="./Images"

base_dir = os.path.join(os.path.dirname(im_dir), 'Images')
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# directory with our training armut pictures
train_armut_dir = os.path.join(train_dir, 'Armut')  

# directory with our training cilek pictures
train_cilek_dir = os.path.join(train_dir, 'Cilek') 

# directory with our training elma_kirmizi pictures
train_elma_kir_dir = os.path.join(train_dir, 'Elma_Kirmizi')  

# directory with our training elma_yesil pictures
train_elma_yes_dir = os.path.join(train_dir, 'Elma_Yesil') 

# directory with our training mandalina pictures
train_mand_dir = os.path.join(train_dir, 'Mandalina')  

# directory with our training muz pictures
train_muz_dir = os.path.join(train_dir, 'Muz') 

# directory with our training portakal pictures
train_port_dir = os.path.join(train_dir, 'Portakal') 

# directory with our training armut pictures
validation_armut_dir = os.path.join(validation_dir, 'Armut')  

# directory with our training cilek pictures
validation_cilek_dir = os.path.join(validation_dir, 'Cilek') 

# directory with our training elma_kirmizi pictures
validation_elma_kir_dir = os.path.join(validation_dir, 'Elma_Kirmizi')  

# directory with our training elma_yesil pictures
validation_elma_yes_dir = os.path.join(validation_dir, 'Elma_Yesil') 

# directory with our training mandalina pictures
validation_mand_dir = os.path.join(validation_dir, 'Mandalina')  

# directory with our training muz pictures
validation_muz_dir = os.path.join(validation_dir, 'Muz') 

# directory with our training portakal pictures
validation_port_dir = os.path.join(validation_dir, 'Portakal') 

# directory with our training armut pictures
test_armut_dir = os.path.join(test_dir, 'Armut')  

# directory with our training cilek pictures
test_cilek_dir = os.path.join(test_dir, 'Cilek') 

# directory with our training elma_kirmizi pictures
test_elma_kir_dir = os.path.join(test_dir, 'Elma_Kirmizi')  

# directory with our training elma_yesil pictures
test_elma_yes_dir = os.path.join(test_dir, 'Elma_Yesil') 

# directory with our training mandalina pictures
test_mand_dir = os.path.join(test_dir, 'Mandalina')  

# directory with our training muz pictures
test_muz_dir = os.path.join(test_dir, 'Muz') 

# directory with our training portakal pictures
test_port_dir = os.path.join(test_dir, 'Portakal') 

BATCH_SIZE = 256
IMG_SHAPE  = 224

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

image_gen_val = ImageDataGenerator(rescale=1./255)

valid_generator = image_gen_val.flow_from_directory(directory=validation_dir,
                                                 batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode="categorical")

image_gen_test = ImageDataGenerator(rescale=1./255)

test_generator = image_gen_test.flow_from_directory(directory=test_dir,
                                                 batch_size=BATCH_SIZE,
                                                 #shuffle=False,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode="categorical")

tf.keras.backend.clear_session()

layer_neurons = [1024, 512, 256, 128, 56, 28, 14]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (IMG_SHAPE, IMG_SHAPE, 3)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
            
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10

history = model.fit_generator(train_generator,
                              epochs=EPOCHS,
                              validation_data=valid_generator)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)


        
##################################
BATCH_SIZE = 64
IMG_SHAPE  = 224

tf.keras.backend.clear_session()

layer_neurons = [256, 128, 56]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (IMG_SHAPE, IMG_SHAPE, 3)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
            
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


for n in range(5):
    plt.subplot(1,5,n+1)
    plt.imshow(test_generator[0][0][n], cmap = plt.cm.binary)
    plt.title(predictions[n])
    plt.axis('off')

#[0][1][x] ilk batchdeki x.inci resmin label'ı(2.argüman label oluyor, ilk argüman batch no)
#[0][0][x] ilk batchdeki x.inci resmin image'ı(2.argüman image oluyor, ilk argüman batch no)
for n in range(5):
    print(test_generator[0][1][n])
    print(predictions[n])
    plt.imshow(test_generator[0][0][n])
    plt.show()    

############################################################


import tensorflow_hub as hub
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224,3))

feature_extractor.trainable = False

tf.keras.backend.clear_session()

model = tf.keras.Sequential([
        feature_extractor])

layer_neurons = [512, 256, 128, 56]

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
            
model.add(tf.keras.layers.Dense(7, activation='softmax'))


model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,
                    callbacks=[early_stopping])

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test=None
test_generator.reset()
test=test_generator
pred=model.predict(test,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

for n in range(30):
    plt.imshow(test[0][0][n])
    plt.title(predictions[n])
    plt.show()  

loss, accuracy = model.evaluate(test_generator,steps=STEP_SIZE_TEST,
verbose=1)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

saved_keras_model_filepath = './model.h5'

model.save(saved_keras_model_filepath)

reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()