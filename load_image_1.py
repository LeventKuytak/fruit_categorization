import glob
from PIL import Image
import numpy as np
import tensorflow as tf

armut = []
for f in glob.iglob("./Armut/*"):
    armut.append(np.asarray(Image.open(f)))

cilek = []
for f in glob.iglob("./Cilek/*"):
    cilek.append(np.asarray(Image.open(f)))

elma_kir = []
for f in glob.iglob("./Elma_Kirmizi/*"):
    elma_kir.append(np.asarray(Image.open(f)))

elma_yes = []
for f in glob.iglob("./Elma_Yesil/*"):
    elma_yes.append(np.asarray(Image.open(f)))

mandalina = []
for f in glob.iglob("./Mandalina/*"):
    mandalina.append(np.asarray(Image.open(f)))

muz = []
for f in glob.iglob("./Muz/*"):
    muz.append(np.asarray(Image.open(f)))

portakal = []
for f in glob.iglob("./Portakal/*"):
    portakal.append(np.asarray(Image.open(f)))

t_armut=[None] * len(armut)
for i in np.arange(0,len(armut)):
    test_image = tf.convert_to_tensor(armut[i], np.int32)
    test_image = tf.image.resize(test_image, (224, 224))
    test_image = tf.cast(test_image, tf.float32)
    test_image /= 255 
    t_armut[i]=test_image

td_armut={}
for i in range(len(t_armut)):
    td_armut[i] = dict([("0",t_armut[i])])

for label, image in td_armut[242].items():
    print(label)
    print(image)



