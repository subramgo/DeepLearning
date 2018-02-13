"""
Preprocess and Load Adience Face Benchmark data into h5py datastore.

Adience dataset was downloaded from 
http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/

The script does the following

1. Pre-process the labels txt file
2. Create a new labels text file with image location and Y labels
3. creates a h5py file to store the images and their labels. 
	a. Images are resized to 256 x 256.


"""

# TODO create easy accessor/generator functions for this data set (`split_by_gender`?)

# TODO parameterize paths for file creation
def create_hdf5():
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
    hdf5_file.create_dataset("train_single_label", (total_images,), np.int8)

    hdf5_file["train_labels"][...] = labels_df[['age_class', 'gender_class']]
    hdf5_file["train_single_label"][...] = labels_df['all_class']

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


def split_by_gender():
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


