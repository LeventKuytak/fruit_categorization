from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from load_image import armut, cilek, elma_kir, elma_yes, mandalina, muz, portakal

armutt = [None] * len(armut)
for i in np.arange(0,len(armut)):
    test_image = tf.convert_to_tensor(armut[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    armutt[i]=test_image.numpy() 

cilekk = [None] * len(cilek)
for i in np.arange(0,len(cilek)):
    test_image = tf.convert_to_tensor(cilek[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    cilekk[i]=test_image.numpy()

elma_kirr = [None] * len(elma_kir)
for i in np.arange(0,len(elma_kir)):
    test_image = tf.convert_to_tensor(elma_kir[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    elma_kirr[i]=test_image.numpy()

elma_yess = [None] * len(elma_yes)
for i in np.arange(0,len(elma_yes)):
    test_image = tf.convert_to_tensor(elma_yes[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    elma_yess[i]=test_image.numpy()

mandalinaa = [None] * len(mandalina)
for i in np.arange(0,len(mandalina)):
    test_image = tf.convert_to_tensor(mandalina[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    mandalinaa[i]=test_image.numpy()

muzz = [None] * len(muz)
for i in np.arange(0,len(muz)):
    test_image = tf.convert_to_tensor(muz[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    muzz[i]=test_image.numpy()

portakall = [None] * len(portakal)
for i in np.arange(0,len(portakal)):
    test_image = tf.convert_to_tensor(portakal[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    portakall[i]=test_image.numpy()

print(armutt.shape, cilekk.shape, elma_kirr.shape, elma_yess.shape, \
      mandalinaa.shape, muzz.shape, portakall.shape)