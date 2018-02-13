from keras.models import load_model
from keras.engine import InputLayer
import cv2
import sys
import numpy as np
import time
import keras.backend as K
from DLUtils.configs import get_configs
from DLUtils.DataGenerator import vgg_train_generator, vgg_eval_generator

K.set_image_data_format('channels_last')

face_dict = {1:"face",0:"noface"}

config_dict = get_configs("vgg")
# Load the model
model_path = config_dict['model_path']
model = load_model(model_path)

#Get an input image from commandline
image_path = sys.argv[1]

#
target_size = config_dict['target_size']
img_w = target_size[0]
img_h = target_size[1]


def gen_test():
	eval_gen = vgg_eval_generator(12)
	print(model.evaluate_generator(eval_gen, steps = 10))

def simple_test(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (img_w,img_h))
	image = image.reshape((1,img_w,img_h,3))
	prediction = model.predict(image, batch_size=1)
	idx = np.argmax(prediction[0])
	print(prediction[0])
	print(face_dict[idx])


def get_pyramids(img, steps = 3):
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
	org_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
	org_img = cv2.resize(org_img,(300,300))
	org_img_shape = org_img.shape
	window_size = (32,32)
	step_size = 8
	all_windows = []
	to_predict = []
	to_windows =[]
	# for different scales
	for img_scale in get_pyramids(org_img):
		for (x,y,window) in sliding_window(img_scale, step_size, window_size):
			if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
				continue
			window = cv2.resize(window, (img_w, img_h))
			pred_input = window.reshape(1,img_w,img_h,3)
			#to_predict.append(pred_input)
			predictions = model.predict(pred_input, batch_size = 1)
			idx = np.argmax(predictions[0])
			if idx == 1:
				all_windows.append([x,y,x + img_w, y + img_h])

	#p_array = np.array(to_predict)
	#p_array = p_array.reshape((len(to_predict), img_w, img_h, 3))
	#predictions = model.predict(p_array, batch_size=len(to_predict))
	#print(predictions)
	#for i, prediction in enumerate(predictions):
	#	idx= np.argmax(prediction)
	#	print(i, prediction, idx, face_dict[idx])
	#	if idx == 1:
	#		all_windows.append([x,y,x + img_w, y + img_h])

	return all_windows


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# initialize the list of picked indexes
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
	return boxes[pick]

def run_pipeline(image_path):
	all_windows = np.array(test_pipeline(image_path))
	print (all_windows.shape)
	selected = non_max_suppression_slow(all_windows, 0.5)
	print(selected)
	window_size = (img_w,img_h)
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.resize(image, (300,300))
	for (x1,y1,x2,y2) in selected:
		image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.imshow("image",image)
		cv2.waitKey(0)

if __name__ == '__main__':
	run_pipeline(image_path)
	#simple_test(image_path)
	#gen_test()
	#config_dict = get_configs('faces12net')
	#gen_test()



