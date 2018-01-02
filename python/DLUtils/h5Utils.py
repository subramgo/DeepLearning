
import glob
import pandas as pd
import os
import cv2
import numpy as np
import h5py


def create_h5_file(csv_file_path, hdf5_path, image_w, image_h,class_col, directory_col,type_col=None):



    labels_df = pd.read_csv(csv_file_path)
    train_df = labels_df[labels_df['type'] == 'train']
    if type_col is not None:
        eval_df = labels_df[labels_df['type'] == 'eval'] 

    total_images = labels_df.shape[0] 
    if type_col is not None:
        eval_images = eval_df.shape[0]
    else:
        eval_images = 0

    train_images = train_df.shape[0]

    print("Train {} Eval {} Total {}".format(train_images, eval_images, total_images))

    ############### h5py file creation #####################
    train_shape = (train_images, image_w, image_h, 3)
    if type_col is not None:
        eval_shape  = (eval_images, image_w, image_h, 3)

    # Open a hdf5 file and create earray
    hdf5_file = h5py.File(hdf5_path, mode = 'w')

    hdf5_file.create_dataset("train_images", train_shape, np.int8)
    hdf5_file.create_dataset("train_labels", (train_images,), np.int8)

    if type_col is not None:
        hdf5_file.create_dataset("eval_images", eval_shape, np.int8)
        hdf5_file.create_dataset("eval_labels", (eval_images,), np.int8)

    hdf5_file["train_labels"][...] = train_df[class_col]

    if type_col is not None:
        hdf5_file["eval_labels"][...] =  eval_df[class_col]



    images_location_list = labels_df[directory_col].tolist()
    type_list = labels_df[type_col].tolist()
    ## Load train image
    i = 0
    train_i = 0
    eval_i = 0
    for (_type, path) in zip(type_list, images_location_list):
        if i % 1000 == 0 and i > 0:
            print("Train data {}/{}".format(i, total_images))
        
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (image_w, image_h)) 
        if type_col is not None:
            if _type == 'train':
            	hdf5_file["train_images"][train_i, ...] = resized_image[None]
            	train_i+=1

            if _type == 'eval':
            	hdf5_file["eval_images"][eval_i, ...] = resized_image[None]
            	eval_i+=1
        else:
            hdf5_file["train_images"][i, ...] = resized_image[None]


        i+=1

    hdf5_file.close()

    print("Finished creating hdf5 file.")

