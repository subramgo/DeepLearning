from flask import Flask,request
import cv2
import numpy as np 
from DLUtils import evaluate
import json

model_path = '../../models/age_gender_model-0.3.h5'
predictor = evaluate.DemographicClassifier(model_path=model_path)

app = Flask(__name__)
app.config['DEBUG'] = False
app.debug = False

@app.route('/gender', methods =['POST'])
def gender():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image_arr, (100, 100)) 
    resized_image = resized_image.reshape(1,100,100,3)
    pred_val = predictor.process(x_test=resized_image)
    return json.dumps({'gender': pred_val})
