
#physical_devices = tf.config.experimental.list_physical_devices('CPU')
#assert len(physical_devices) > 0, "Not enough CPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

app= Flask(__name__)

base_path = os.path.dirname(__file__)

MODEL_PATH = os.path.join(base_path,'model/detectPneumonia98psr.json')
MODEL_WEIGHT_PATH = os.path.join(base_path,'model/detectPneumonia98psr.h5')

with open(MODEL_PATH, 'r') as f:
    json_model_file = f.read()

model = model_from_json(json_model_file)
model.load_weights(MODEL_WEIGHT_PATH)

model.compile(optimizer = RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

model._make_predict_function()
print("Loaded model from disk")


def binary_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    preds = model.predict(images, batch_size=10)
    if(preds[0]>0):
        return True
    else:
        return False

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if(request.method=='POST'):
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 
                                'user_uploads',
                                secure_filename(f.filename))
        f.save(file_path)
        result = binary_model_predict(file_path,model)

        if(result==True):
            return "Pneumonia test result is positive"
        else:
            return "Pneumonia test result is negative"
    return None


#app.run(debug=True)
if __name__== '__main__':
    app.run(debug=True)
