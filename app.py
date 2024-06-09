import threading

from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

app = Flask(__name__)

dropbox_url = "https://www.dropbox.com/scl/fi/6dy5g2m3yuso5ru9v3xax/modelo_ecografia.h5?rlkey=u52kwi9jlqf6s63euglx21p2y&st=lwqirnmj&dl=1"

# Ruta para guardar el modelo descargado
model_path = 'model/modelo_ecografia.h5'
model = None
model_loaded = False

def download_and_load_model():
    global model, model_loaded
    if not os.path.exists(model_path):
        response = requests.get(dropbox_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    model = load_model(model_path)
    model_loaded = True

# Descargar y cargar el modelo en un hilo separado
threading.Thread(target=download_and_load_model).start()

#####################

# Cargar el modelo
#model = load_model('model/modelo_ecografia.h5')
img_sz = 300
class_labels = {0: 'Benigna', 1: 'Maligna', 2: 'Normal'}  # clases del modelo

predefined_images = {
    'example1': 'static/benign (5).png',
    'example2': 'static/malignant (99).png',
    'example3': 'static/normal (79).png'
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        selected_example = request.form.get('example')
        if selected_example:
            filepath = predefined_images[selected_example]
        else:
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                filepath = os.path.join('static', file.filename)
                file.save(filepath)

        img = image.load_img(filepath, target_size=(img_sz, img_sz), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)

        predicted_label = class_labels[predicted_class[0]]

        return render_template('result.html', image_file=filepath, result=predicted_label, confidence=confidence)
    return render_template('index.html')

@app.route('/model-status')
def model_status():
    return jsonify({"model_loaded": model_loaded})

if __name__ == '__main__':
    app.run(debug=True)
