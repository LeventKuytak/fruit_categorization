import glob
from PIL import Image
import numpy as np
import tensorflow as tf

armut = []
for f in glob.iglob("./Armut/*"):
    armut.append(np.asarray(Image.open(f)))

armut = np.array(armut)

cilek = []
for f in glob.iglob("./Cilek/*"):
    cilek.append(np.asarray(Image.open(f)))

cilek = np.array(cilek)

elma_kir = []
for f in glob.iglob("./Elma_Kirmizi/*"):
    elma_kir.append(np.asarray(Image.open(f)))

elma_kir = np.array(elma_kir)

elma_yes = []
for f in glob.iglob("./Elma_Yesil/*"):
    elma_yes.append(np.asarray(Image.open(f)))

elma_yes = np.array(elma_yes)

mandalina = []
for f in glob.iglob("./Mandalina/*"):
    mandalina.append(np.asarray(Image.open(f)))

mandalina = np.array(mandalina)

muz = []
for f in glob.iglob("./Muz/*"):
    muz.append(np.asarray(Image.open(f)))

muz = np.array(muz)

portakal = []
for f in glob.iglob("./Portakal/*"):
    portakal.append(np.asarray(Image.open(f)))

portakal = np.array(portakal)

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

n_armut=np.asarray(armutt)
t_armut=tf.convert_to_tensor(n_armut)

n_cilek=np.asarray(cilekk)
t_cilek=tf.convert_to_tensor(n_cilek)

n_elma_kir=np.asarray(elma_kirr)
t_elma_kir=tf.convert_to_tensor(n_elma_kir)

n_elma_yes=np.asarray(elma_yess)
t_elma_yes=tf.convert_to_tensor(n_elma_yes)

n_mandalina=np.asarray(mandalinaa)
t_mandalina=tf.convert_to_tensor(n_mandalina)

n_muz=np.asarray(muzz)
t_muz=tf.convert_to_tensor(n_muz)

n_portakal=np.asarray(portakall)
t_portakal=tf.convert_to_tensor(n_portakal)