from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the CNN model
model = load_model('model/model_cnn.h5')

# Target image size (change if your model uses a different size)
IMAGE_SIZE = (224, 224)

# Class labels (update according to your trained model)
classes = ['Floral', 'Geometric', 'Striped', 'Abstract']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/discover')
def discover():
    return render_template('discover.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('home.html', prediction_text="No file uploaded.")

    file = request.files['file']

    if file.filename == '':
        return render_template('home.html', prediction_text="No selected file.")

    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize if model was trained this way
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)

        predicted_class = classes[np.argmax(prediction)]

        return render_template('home.html',
                               prediction_text=f'Predicted Class: {predicted_class}',
                               img_path=filepath)

    return render_template('home.html', prediction_text="Something went wrong.")

if __name__ == '__main__':
    app.run(debug=True)
