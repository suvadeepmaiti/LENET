import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

root = os.getcwd()
print(root)
sys.path.append(os.path.join(root, 'src', 'scripts'))
# print(os.path.join(root, '/scripts'))

from lenet import Lenet_SMAI

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

model_path = os.path.join(root, 'src' , 'notebooks' ,'model-relu-tanh-28-512-adam.pickle')


def load_weights(model, weights):
    for i,layer in enumerate(model.layers):
        print(layer) 
        layer.params = weights[i]

# normalization of the input images
def normalize(image):
    image -= image.min()
    image = image / image.max()
    # range = [-0.1,1.175]   
    image = image * 1.275 - 0.1
    return image

# Use pickle to load in the pre-trained model.
with open(model_path, 'rb') as f:
        model = Lenet_SMAI()
        weights = pickle.load(f)
        load_weights(model, weights)

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')



# #Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        resized = normalize(cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA))
        # plt.imsave('ex.jpeg', resized)
        pred = model(resized.reshape(1,32,32,1) , mode = 'test')
        print(pred)
        return render_template('results.html', prediction = {'pred' : pred})


if __name__ == '__main__':
	app.run(debug=True)