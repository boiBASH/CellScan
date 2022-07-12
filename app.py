#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 18:27:16 2022

@author: mac
"""
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
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)


MODEL_PATH = 'models/Malaria_cell_classifation.h5'

#Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
      
print('Model loaded. Start serving...')

def prepare_image(image, target):
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)


	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
			# read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
            image = prepare_image(image, target=(50, 50))

            preds = model.predict(image)
            pred = np.argmax(preds,axis = 1)
            str1 = 'Malaria Parasite Present'
            str2 = 'Normal'
            if pred[0] == 0:
               return str1
            else:
               return str2

	# return the data dictionary as a JSON response
    return flask.jsonify(pred)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
    
    app.debug = True
    app.run(host='0.0.0.0')



