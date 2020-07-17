import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image


import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt



physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


test_dir = 'test'
test_fnames = os.listdir(test_dir)


with open('models/detectPneumonia98psr.json', 'r') as f:
    json_model_file = f.read()

model = tf.keras.models.model_from_json(json_model_file)
model.load_weights("models/detectPneumonia98psr.h5")
print("Loaded model from disk")

model.compile(optimizer = RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])


test_set = test_fnames[:600]

mistakes = 0
count= 0
for fn in test_set:
    count=count+1
    path = os.path.join(test_dir, fn)
    img= image.load_img(path, target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])


    if(classes[0]>0):
        print(fn + " is positive")
    else:
        print(fn + " is negative")
        mistakes = mistakes + 1



print("total: ",count)
print("mistakes: ", mistakes)
accuracy=((count-mistakes)/count)
print("accuracy: ", accuracy)
