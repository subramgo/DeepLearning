"""
    Interface providing data sources from local files such as H5Py
"""

import numpy as _np
import h5py as _h5py
from keras.utils import np_utils as _np_utils

def _generatorFactory(filepath,x_label='train_images',y_label='train_labels'):
    """
        Produces a generator function.
        Give a filepath and column labels:
            HDF5 data is loaded from the filepath
            x and y labels need to match the HDF5 file's columns
    """
    def _generator(dimensions,nbclasses,batchsize):
        while 1:
            with _h5py.File(filepath, "r") as f:
                filesize = len(f[y_label])
                n_entries = 0
                while n_entries < (filesize - batchsize):
                    x_train= f[x_label][n_entries : n_entries + batchsize]
                    x_train= _np.reshape(x_train, (batchsize, dimensions[0], dimensions[1], 3)).astype('float32')

                    y_train = f[y_label][n_entries:n_entries+batchsize]
                    # data-specific formatting should be done elsewhere later, even onecoding
                    # if dimensions is needed, can be gotten from x_train.shape
                    y_train_onecoding = _np_utils.to_categorical(y_train, nbclasses)

                    n_entries += batchsize

                    # Shuffle
                    p = _np.random.permutation(len(y_train_onecoding))
                    yield (x_train[p], y_train_onecoding[p])
                f.close()
    
    return _generator



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

# TODO make script to download this data to a local path automatically if it isn't present already
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

#    config = configs.Config()
#    _filepath = config.get_section_dict('gender_resnet')['h5_path']
#    _factory = _generatorFactory(_filepath)
#    adience_train_generator = _factory(batchsize=3)
#
#
#    #######################################
#    ###     Adience Data Generators     ###
#    #######################################
#    _filepath = config.get_section_dict('adience')['data_path']
#
#
#    _adience_train_factory = _generatorFactory(_filepath)
#    adience_train_generator = _adience_train_factory(batchsize=3)
#
#
#    _adience_eval_factory = _generatorFactory(_filepath,x_label='eval_images',y_label='eval_labels')
#    adience_eval_generator = _adience_eval_factory(batchsize=3)

