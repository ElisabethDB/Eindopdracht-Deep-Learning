import flask
from PIL import Image
import numpy as np
from tensorflow import keras
from flask import Flask, render_template


# Create a Flask app
app = Flask(__name__)

# Load the prediction model
model = keras.models.load_model('modellen/Xception_fine_tuning.keras')


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = flask.request.files.get('image')

    # Load the image into a PIL Image object
    img = Image.open(image_data).convert('RGB')

    img_new = img.resize((180, 180))  # Resize the image to 180*180
    img_new = np.array(img_new) / 255.0  # Normalize the pixel values

    # Make a prediction on the image
    predictions = model.predict(np.expand_dims(img_new, axis=0))

    # Determine the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Map the integer class label to a string label

    categories = ['Picasso','Rubens','Mondriaan']
    predicted_label = categories[predicted_class]

    # Return the prediction as a JSON response
    return render_template('index.html', pred=str(predicted_label))


if __name__ == '__main__':
    app.run(debug=True)