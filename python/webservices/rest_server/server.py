from flask import Flask,request
import cv2
import numpy as np 
from DLUtils import evaluate
import json

predictor = evaluate.GenderClassifier()

app = Flask(__name__)
app.config['DEBUG'] = False
app.debug = False

@app.route('/gender', methods =['POST'])
def gender():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred_val = predictor.process(x_test=image_arr)
    return json.dumps({'gender': pred_val})
