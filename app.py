from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np



# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Malaria_cell_classifation.h5'

#Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()      
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    img = image.load_img("uploads/C101P62ThinF_IMG_20150918_151507_cell_63.png", target_size=(50,50)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        
        str1 = 'Malaria Parasite Present'
        str2 = 'Normal'
        if pred[0] == 0:
            return str1
        else:
            return str2
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':

        app.run(host='0.0.0.0')
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
        #http_server = WSGIServer(('', 5000), app)
        #http_server.serve_forever()
