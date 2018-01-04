from keras.models import load_model
from keras.engine import InputLayer
import cv2
import sys
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras.backend as K
from DLUtils.configs import get_configs
from DLUtils.DataGenerator import face_12net_train_generator, face_12net_eval_generator

K.set_image_data_format('channels_last')


model_path = sys.argv[1]


# Load the model
model = load_model(model_path)

#Get an input image from commandline
image_path = sys.argv[2]

image_w = int(sys.argv[3])
image_h = int(sys.argv[4])

def simple_test(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (image_w,image_h))
	image = image.reshape((1,image_w,image_h,3))
	prediction = model.predict(image, batch_size=1)
	print(prediction.shape)
	idx = np.argmax(prediction[0])
	print(prediction[0])
	print(idx)

if __name__ == "__main__":
	simple_test(image_path)
