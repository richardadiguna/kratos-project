import os
import io
import cv2
import json
import base64
import requests
import numpy as np

from bunch import Bunch
from flask import Flask
from flask import request
from flask import jsonify

UPLOAD_FOLDER = '/temp'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

HOST = 'localhost:8501'
MODEL_NAME = 'kratos'
VERSION = 1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_prediction_from_model(img):

    if img.shape != (128, 128):
        img = cv2.resize(img, (128, 128))

    payload = {"instances": [{'images': img.tolist()}]}
    r = requests.post(
        'http://localhost:8501/v1/models/kratos:predict',
        json=payload)
    b = r.content.decode('utf8').replace("'", '"')
    data = json.loads(b)
    response = Bunch(data)

    rank = np.argmax(response.predictions)

    if rank == 0:
        return 'background'
    elif rank == 1:
        return 'e-ktp'
    elif rank == 2:
        return 'ktp'
    elif rank == 3:
        return 'dukcapil'

    return 'undefined'


def convert_im_file(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(
        in_memory_file.getvalue(),
        dtype=np.uint8)
    color_image_flag = 1
    return cv2.imdecode(data, color_image_flag)


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/prediction", methods=['POST'])
def get_prediction():
    if request.method == 'POST':

        if 'image' not in request.files:
            print('No file part, key: image')
            return jsonify({"error": "No file part, key: image"})

        file = request.files['image']

        if file.filename == '':
            print('No selected file')
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            img_decoded = convert_im_file(file)
            result = get_prediction_from_model(img_decoded)
            return jsonify({"result": result})

    return jsonify({"error": "500"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
