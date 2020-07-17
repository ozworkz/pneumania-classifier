import os
import numpy as np
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img



physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


base_dir = './'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training normal/pneumonia pictures
train_normal_dir = os.path.join(train_dir, 'NORMAL')
train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

# Directory with our validation normal/pneumonia pictures
validation_normal_dir = os.path.join(validation_dir, 'NORMAL')
validation_pneumonia_dir = os.path.join(validation_dir, 'PNEUMONIA')

train_normal_fnames = os.listdir( train_normal_dir )
train_pneumonia_fnames = os.listdir( train_pneumonia_dir )

#test fnames
test_fnames = os.listdir(test_dir)

nrows = 4
ncols = 4
pic_index = 20

fig = plt.gcf()
fig.set_size_inches(ncols * 4 , nrows * 4)
pic_index += 8

next_normal_pix = [os.path.join(train_normal_dir, fname) for fname in train_normal_fnames[pic_index-8:pic_index]]
next_pneumonia_pix = [os.path.join(train_pneumonia_dir, fname) for fname in train_pneumonia_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_normal_pix + next_pneumonia_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')
    img= mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

class saveEpochModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch = 0
        self.modelname = 'detectPneumonia'
        self.modeldir = 'models'
        self.modelpath = os.path.join(self.modeldir, self.modelname)

    def on_epoch_end(self, epoch, logs={}):
        model_json = self.model.to_json()
        with open("{}{}.json".format(self.modelpath,str(self.epoch).zfill(4)), "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights("{}{}.h5".format(self.modelpath,str(self.epoch).zfill(4)))
        print("\nSaved {} model to disk\n".format(self.modelname))

        self.epoch += 1

callbacks =saveEpochModelCallback()

model.compile(optimizer = RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

train_datagen = ImageDataGenerator(
    rescale = 1.0/ 255.)
#    rotation_range=15,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    zoom_range=0.4,
#    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale= 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
          #                                          color_mode='grayscale',
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))

validation_generator = test_datagen.flow_from_directory(validation_dir,
          #                                              color_mode='grayscale',
                                                        batch_size = 20,
                                                        class_mode = 'binary',
                                                        target_size = (150,150))

history = model.fit_generator(train_generator,
                              validation_data= validation_generator,
                              epochs=10,
                              verbose = 1, callbacks=[callbacks])

test_set = test_fnames[:200]

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
        print(fn + "is pneumonia")
    else:
        print(fn + "is normal")
        mistakes=mistakes+1

print("total: ",count)
print("mistakes: ", mistakes)
accuracy=((count-mistakes)/count)
print("accuracy: ", accuracy)

model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("models/model.h5")
print("Saved model to disk")
