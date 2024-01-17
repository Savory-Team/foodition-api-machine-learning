import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gdown
import joblib
import jwt
import numpy as np
import pandas as pd
from functools import wraps
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from zipfile import ZipFile 

gdown.download('https://drive.google.com/uc?id=1cRiJkt9RtHZB-qITlGobWpe81h7w-phz', output='model.zip', quiet=False)
with ZipFile('./model.zip', 'r') as modelFolder: 
    modelFolder.extractall()

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_EDIBLE_CLASSIFICATION'] = './model/edible_food_classification.h5'
app.config['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/storage-admin-key.json'

model_edible_classification = load_model(app.config['MODEL_EDIBLE_CLASSIFICATION'], compile=False)

bucket_name = os.environ.get('BUCKET_NAME')
client = storage.Client.from_service_account_json(json_credentials_path=app.config['GOOGLE_APPLICATION_CREDENTIALS'])
bucket = storage.Bucket(client, bucket_name)

SECRET_KEY = os.environ.get('SECRET_KEY')
if SECRET_KEY is None:
    print("SECRET_KEY not found in environment variables.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization', None)
        if not token:
            return jsonify({'message': 'Invalid token'}), 401
        try:
            token_prefix, token_value = token.split()
            if token_prefix.lower() != 'bearer':
                raise ValueError('Invalid token prefix')
            data = jwt.decode(token_value, SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        except ValueError:
            return jsonify({'message': 'Invalid token format'}), 401
        return f(data, *args, **kwargs)
    return decorator

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ',
        'Data': {
            'Project': 'Fodition',
            'Team': 'Savory',
            'Moto': 'Model yang baik adalah model yang dapat memprediksi cinta mu, awwwwww pengen ngising',
            'CreatedBy': 'Aditya Bayu & Novebri Tito',
            'Copyright': '©2024 All Rights Reserved!'
        }
    }), HTTPStatus.OK

@app.route('/predict', methods=['POST'])
@token_required
def predict_edible_classification(data):
    if data is None:
        return jsonify({
            'status': {
                'code': HTTPStatus.FORBIDDEN,
                'message': 'Akses dilarang',
            }
        }), HTTPStatus.FORBIDDEN
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((160, 160))
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255
            predicted_class = None
            classificationResult = model_edible_classification.predict(x, batch_size=1)
            class_list = ['apple - edible', 'apple - inedible', 'banana - edible', 'banana - inedible', 'bread - edible', 'bread - inedible', 'chicken - edible', 'chicken - inedible', 'donut - edible', 'donut - inedible', 'edible', 'egg - edible', 'egg - inedible', 'mango - edible', 'mango - inedible', 'orange - edible', 'orange - inedible', 'pizza - edible', 'pizza - inedible', 'potato - edible', 'potato - inedible', 'rice - edible', 'rice - inedible', 'tomato - edible', 'tomato - inedible']
            predicted_class = class_list[np.argmax(classificationResult[0])]
            result = (lambda x : 'edible' if x == 'edible' else predicted_class.split(' - ')[1])(predicted_class)
            image_name = image_path.split('/')[-1]
            blob = bucket.blob('food-images/' + image_name)
            blob.upload_from_filename(image_path) 
            os.remove(image_path)
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'class': result }
                }
            }), HTTPStatus.OK
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
