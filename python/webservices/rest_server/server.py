import sys
import keras
from keras import optimizers

from flask.views import MethodView
import json

from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import numpy as np

class Predictor(object):
    """Provides predict() while abstracting away model and preprocessing"""
    #TODO add preprocessing to _normalize_input and _denormalize_prediction
        
    def __init__(self, model_path, **kwargs):
        """Initialize Predictor class."""
        self.model = keras.models.load_model(model_path)
        
        sgd = optimizers.SGD(lr=0.001, momentum=0.95, decay=1e-6, nesterov=False)
        self.model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ["accuracy"])

    def _normalize_input(self, X_input):
        """No normalization performed"""
        return X_input

    def _denormalize_prediction(self, x_pred):
        """No De-normalization performed"""
        return x_pred

    def predict(self, X_input):
        """Make predictions, given some input data.

        This normalizes the predictions based on the real normalization
        parameters and then generates a prediction

        Args:
            X_input:    Input vector to for prediction
        """
        x_normed = self._normalize_input(X_input=X_input)
        x_pred = self.model.predict(x_normed)
        prediction = self._denormalize_prediction(x_pred)
        return prediction


class ModelLoader(MethodView):
    """Flask-friendly class. Initialize model params and wait for a POST request"""

    def __init__(self):
        model_path = '../models/age_gender_model_0_1.h5'
        self.predictor = self.initialize_models(model_path=model_path)

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
    
    def initialize_models(self,model_path):
        """Initialize models and use this in Flask server."""
        predictor = Predictor(model_path)
        return predictor

def run(host='0.0.0.0', port=7171):
    app = Flask(__name__)
    app.add_url_rule('/predict', view_func=ModelLoader.as_view('predict'))
    print('running server http://{0}'.format(host + ':' + str(port)))
    WSGIServer((host, port), app).serve_forever()


run()



