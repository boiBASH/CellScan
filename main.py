from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
import io

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


def prepare_image(image, target):
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)


	# return the processed image
	return image

# Model saved with Keras model.save()
MODEL_PATH = 'models/Malaria_cell_classifation.h5'

#Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()      
print('Model loaded. Start serving...')

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def model_predict(img_path, model):
    img_path = convertToBinaryData(img_path.filename)

    image = Image.open(io.BytesIO(img_path))
    #img = image.load_img(img_path, target_size=(50,50)) #target_size must agree with what the trained model expects!!
    
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(50, 50))
    
    # Preprocessing the image
    #img = image.img_to_array(img)
    #img = np.expand_dims(img, axis=0)

   
    preds = model.predict(image)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        #return redirect(url_for('index'))
    

        # Make prediction
    pred = model_predict(uploaded_file, model)
    str1 = 'Malaria Parasite Present'
    str2 = 'Normal'
    if pred[0] == 0:
        return str1
    else:
        return str2

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':

        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
        #http_server = WSGIServer(('', 5000), app)
        #http_server.serve_forever()
