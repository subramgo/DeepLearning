import glob
import pandas as pd
import os
import cv2
import numpy as np
from shutil import copy2

image_path  = '../data/Adience/faces/'

male_dest   = '../data/Adience/gender/train/male/'
female_dest ='../data/Adience/gender/train/female/'
# Labels folder
fnl_labels_path = '../data/Adience/'
labels_df = pd.read_csv(fnl_labels_path + 'labels.csv')


images_location_list = labels_df['image_loc'].tolist()
gender_list = labels_df['gender'].tolist()


for gender, file in zip(gender_list, images_location_list):
	if gender == 'm':
		copy2(file, male_dest)
	elif gender == 'f':
		copy2(file, female_dest)

print("done")
