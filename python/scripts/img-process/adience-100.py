"""
Preprocess and Load Adience Face Benchmark data into h4py datastore.

Adience dataset was downloaded from 
http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/

The script does the following

1. Pre-process the labels txt file
2. Create a new labels text file with image location and Y labels
3. creates a h5py file to store the images and their labels. 
	a. Images are resized to 256 x 256.


"""
import glob
import pandas as pd
import os
import cv2
import numpy as np
import h5py


image_path  = '../../data/Adience/faces/'

# Labels folder
fnl_labels_path = '../data/Adience/'
hdf5_path = '../data/Adience/hdf5/adience-100.h5'
labels_df = pd.read_csv(fnl_labels_path + 'labels.csv')

total_images = labels_df.shape[0] 

############### h5py file creation #####################

train_shape = (total_images, 100, 100, 3)

# Open a hdf5 file and create earray
hdf5_file = h5py.File(hdf5_path, mode = 'w')

hdf5_file.create_dataset("train_images", train_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (total_images,2), np.int8)
hdf5_file["train_labels"][...] = labels_df[['age_class', 'gender_class']]

mean = np.zeros(train_shape[1:], np.float32)
images_location_list = labels_df['image_loc'].tolist()
## Load train image
i = 0
for path in images_location_list:
    if i % 1000 == 0 and i > 0:
        print("Train data {}/{}".format(i, total_images))
    
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (100, 100)) 
    hdf5_file["train_images"][i, ...] = resized_image[None]
    mean += resized_image / float(total_images)
    i+=1

hdf5_file["train_mean"][...] = mean
hdf5_file.close()

print("Finished creating hdf5 file.")



