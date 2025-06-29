from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)


# Load the trained model
model = load_model('model/model_cnn.h5')

# Mapping class indices
class_names = ['animal', 'cartoon', 'floral', 'geometry', 'ikat', 'plain', 'polka dot', 'squares', 'stripes', 'tribal']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/discover')
def discover():
    return render_template('discover.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        result = class_names[class_index]

        return render_template('result.html', prediction=result, user_image=filepath)

if __name__ == '__main__':
    app.run(debug=True)
