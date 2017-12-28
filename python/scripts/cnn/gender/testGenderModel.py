import numpy as np 
from keras.models import Model
import h5py
import sys
import os
from DLUtils.evaluate import GenderClassifier
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

eval = GenderClassifier()


if process_multiple:
	f = open('results.txt', 'w')

	file_list = glob.glob(image_path + '*.jpg')
	for file in file_list:
		print(file)
		image = cv2.imread(file, cv2.IMREAD_COLOR)
		gender = eval.process(image, None, 1)
		f.write(str(file) + ',' + str(gender) + '\n')
		gender = eval.process(resized_image)
		f.write(str(file) + ',' + str(gender) + '/n')

	f.close()

else:
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	gender = eval.process(image)
	print(str(gender))




