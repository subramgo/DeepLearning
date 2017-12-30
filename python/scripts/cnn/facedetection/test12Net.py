from keras.models import load_model
from keras.engine import InputLayer
import cv2
import sys
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras.backend as K
from DLUtils.configs import get_configs

K.set_image_data_format('channels_last')



# Load the model
model_path = '../cellar/face12net_1.h5'
model = load_model(model_path)

#Get an input image from commandline
#image_path = sys.argv[1]


def gen_test():
	test_datagen = ImageDataGenerator(rescale=1./255)
	validation_generator = test_datagen.flow_from_directory(
            config_dict['eval_path'],
            target_size=config_dict['target_size'],
            batch_size=config_dict['batch_size'],
            classes =['face','noface'],
            class_mode='categorical')
	predictions = model.predict_generator(validation_generator, steps = 1)
	print(predictions)

def simple_test(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (12,12))
	image = image * (1.0/255)
	image = image.reshape((1,12,12,3))
	prediction = model.predict(image, batch_size=1)
	idx = np.argmax(prediction[0])
	print(prediction[0])
	print(idx)


def get_pyramids(img, steps = 6):
	# https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
	a = img.copy()
	yield a
	for i in range(steps):
		a = cv2.pyrDown(a)
		yield a

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
		

def test_pipeline(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	window_size = (12,12)
	step_size = 4
	all_windows = []
	# for different scales
	for img_scale in get_pyramids(image):
		for (x,y,window) in sliding_window(img_scale, step_size, window_size):
			if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
				continue

			pred_input = window.reshape(1,12,12,3)
			prediction = model.predict(pred_input, batch_size=1)
			idx = np.argmax(prediction)
			if idx == 1 and prediction[0][idx] > 0.6:
				all_windows.append((x,y,window,idx,prediction[0][idx]))

			#clone = window.copy()
			#cv2.rectangle(clone, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
			#cv2.imshow("image", clone)
			#cv2.waitKey(1)
			#time.sleep(0.025)
			#cv2.imwrite(str(x)+":"+str(y)+".jpg", clone)
	return all_windows


def run_pipeline(image_path):
	#print(model.to_json())
	#big_pic(image_path)
	all_windows = test_pipeline(image_path)
	window_size = (12,12)
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	for (x,y,_,_,_) in all_windows:
		image = cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
	cv2.imshow("image",image)
	cv2.waitKey(0)

if __name__ == '__main__':
	#simple_test(image_path)
	config_dict = get_configs('faces12net')
	gen_test()



