import numpy as np 
from keras.models import Model
import h5py
import sys
from utils.evaluate import Evaluate




model_path = '../models/age_gender_model-0.3.h5'
image_path = sys.argv[1]


image = cv2.imread(image_path, cv2.IMREAD_COLOR)
resized_image = cv2.resize(image, (100, 100)) 

eval = Evaluate(model_path, resized_image, None, batch_size = 1)
eval.process(eval = False, predict = True)



