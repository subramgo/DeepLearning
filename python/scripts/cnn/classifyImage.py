import numpy as np 
from keras.models import Model
import h5py
import sys
import sys,os
sys.path.append(os.getcwd())
from utils.evaluate import Evaluate
import cv2
import glob

image_path = sys.argv[1]

process_multiple = False 

if os.path.isdir(image_path):
	process_multiple = True 
elif os.path.isfile(image_path):
	process_multiple = False
else:
	print("Invalid path")
	exit()

model_path = '../models/age_gender_model-0.3.h5'
eval = Evaluate(model_path)

f = open('results.txt', 'w')

if process_multiple:
	file_list = glob.glob(image_path + '*.jpg')
	for file in file_list:
		print(file)
		image = cv2.imread(file, cv2.IMREAD_COLOR)
		resized_image = cv2.resize(image, (100, 100)) 
		resized_image = resized_image.reshape(1,100,100,3)
		gender = eval.process(resized_image, None, 1, eval = False, predict = True)
		f.write(str(file) + ',' + str(gender) + '/n')

	f.close()

else:
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	resized_image = cv2.resize(image, (100, 100)) 
	resized_image = resized_image.reshape(1,100,100,3)

	gender = eval.process(resized_image, None, 1,eval = False, predict = True)
	print(str(file) + ',' + str(gender))




