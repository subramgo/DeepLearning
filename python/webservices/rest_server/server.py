import sys
import keras
from keras import optimizers
import numpy as np

from DLUtils import Evaluation

from flask import Flask, request, jsonify
from flask.views import MethodView
import json
from gevent.pywsgi import WSGIServer

class Predictor(object):
    """Provides predict() while abstracting away model and preprocessing"""
    #TODO add preprocessing to _normalize_input and _denormalize_prediction
    
    def __init__(self, model_path, **kwargs):
        """Initialize Predictor class."""
        self.predictor = Evaluation.DemographicClassifier(model_path)

    def predict(self, X_input):
        """Make predictions, given some input data.

        This normalizes the predictions based on the real normalization
        parameters and then generates a prediction

        Args:
            X_input:    Input vector to for prediction
        """
        prediction = self.predictor.process(X_input)
        return prediction


class ModelLoader(MethodView):
    """Flask-friendly class. Initialize model params and wait for a POST request"""

    def __init__(self):
        model_path = '../models/age_gender_model_0_1.h5'
        self.predictor = Evaluation.DemographicClassifier(model_path=model_path)

    def post(self):
        """Accept a post request to serve predictions."""
        content = request.get_json()
        X_input = content['X_input']
        dimensions = content['dimensions']
        X_input = np.reshape(np.array(json.loads(X_input)),dimensions)
        if not isinstance(X_input, np.ndarray):
            X_in = np.reshape(np.array(X_input), dimensions)
        pred_val = self.predictor.predict(X_input=X_input)
        pred_val = pred_val.tolist()
        return jsonify({'pred_val': pred_val})
    
    def get(self):
        pass
        #TODO similar to above POST

def run(host='0.0.0.0', port=7171):
    app = Flask(__name__)
    app.add_url_rule('/demographics/<model_id>/image/<image_id>', view_func=ModelLoader.as_view('predict'))
    
    print('running server http://{0}'.format(host + ':' + str(port)))
    
    WSGIServer((host, port), app).serve_forever()


run()



