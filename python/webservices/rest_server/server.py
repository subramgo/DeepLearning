from flask import Flask,request,render_template,Response,  abort, render_template_string, send_from_directory
import cv2
import numpy as np 
from DLUtils import evaluate # import GenderClassifier
import json
import os
from PIL import Image
from io import StringIO

predictor = evaluate.GenderClassifier()
face = evaluate.FaceRecog()

WIDTH = 1000
HEIGHT = 800


app = Flask(__name__)
#app.config['DEBUG'] = False
#app.debug = False


@app.route('/gender', methods =['POST'])
def gender():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred_val = predictor.process(x_test=image_arr, y_test=None, batch_size=1)
    return json.dumps({'gender': pred_val})

@app.route('/facerecog', methods =['POST'])
def facerecog():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    pred_val =  face.predict(image_arr)
    return json.dumps({'person': pred_val})

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)


@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)

@app.route('/viewimages')
def index():
    images = []
    for root, dirs, files in os.walk("/home/dlftp/ftp/bocacafepvm/"):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg'):
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })

    return render_template("viewimages.html", **{
        'images': images
})
