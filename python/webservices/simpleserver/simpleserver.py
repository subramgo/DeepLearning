from flask import request
import cv2
import numpy as np 
from DLUtils import evaluate

model_path = '../../models/age_gender_model-0.3.h5'
predictor = evaluate.DemographicClassifier(model_path=model_path)



@app.route('/gender', methods =['POST'])
def gender():
	data = request.data
	nparr = np.fromstring(data, np.uint8)
	image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	resized_image = cv2.resize(image_arr, (100, 100)) 
	resized_image = resized_image.reshape(1,100,100,3)
	pred_val = predictor.process(x_test=resized_image)
	return jsonify({'gender': pred_val})