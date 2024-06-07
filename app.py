
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
#####################

from glob import glob

# Directorio donde se guardan las partes
parts_dir = 'model/parts'

# Combinar las partes en un solo archivo
combined_model_path = 'model/modelo_ecografia_reconstruido.h5'

with open(combined_model_path, 'wb') as combined_file:
    for part_file_path in sorted(glob(os.path.join(parts_dir, 'part_*.h5'))):
        with open(part_file_path, 'rb') as part_file:
            combined_file.write(part_file.read())

# Luego, cargar el modelo desde el archivo reconstruido
model = load_model(combined_model_path)

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


if __name__ == '__main__':
    app.run(debug=True)
