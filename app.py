
import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import pickle
import io

app = Flask(__name__)

class FastNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model, label_encoder = pickle.load(file)
    return model, label_encoder

model, label_encoder = load_model("model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1.html')
def index1():
    return render_template('index1.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/display.html')
def display():
    label = request.args.get('label')
    image_data = request.args.get('image_data')
    return render_template('display.html', label=label, image_data=image_data)

def process_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB').resize((128, 128))
    return np.array(image).flatten() / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        img_array = process_image(image_file.read())
        prediction = model.predict(img_array[np.newaxis, ...])
        predicted_species = label_encoder.inverse_transform(prediction)[0]
        return jsonify({'predicted_species': predicted_species})
    except Exception as e:
        return jsonify({'error': str(e)}), 500





