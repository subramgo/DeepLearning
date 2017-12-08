"""
Preprocess and Load Adience Face Benchmark data into h4py datastore.

Adience dataset was downloaded from 
http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/

The script does the following

1. Pre-process the labels txt file
2. Create a new labels text file with image location and Y labels
3. creates a h5py file to store the images and their labels. 
	a. Images are resized to 256 x 26.


"""
import glob
import pandas as pd
import os
import cv2
import numpy as np
import h5py



# Images folder
image_path  = '../../data/Adience/faces/'
# Labels folder
labels_path = '../../data/Adience/*.txt'
fnl_labels_path = '../../data/Adience/'
# A single image to get height and width
single_image = image_path + "30601258@N03/coarse_tilt_aligned_face.1.10399863183_a04f4c26a1_o.jpg"

hdf5_path = '../../data/Adience/hdf5/adience.h5'

# Filters
age_filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}
gen_filter = {'m':0,'f':1}



labels_files = glob.glob(labels_path)

# Read the image paths and labels
#images  = glob.glob(image_path + 'train/' + image_files)

df_from_each_file = (pd.read_table(f, sep='\t') for f in labels_files)
labels_df   = pd.concat(df_from_each_file, ignore_index=True)

def print_stats(labels_df):
	"""
	"""
	print(labels_df.columns.values.tolist())
	print("Print total rows {}".format(labels_df.shape[0] ))
	print(labels_df.groupby(['gender']).size())
	print(labels_df.groupby(['age']).size())

def apply_filters(labels_df):
	"""
	"""
	labels_df = labels_df[labels_df['age'].isin(age_filter.keys())]
	labels_df = labels_df[labels_df['gender'].isin(gen_filter.keys())]
	return labels_df

print("Before Pre-processing")
print_stats(labels_df)
labels_df = apply_filters(labels_df)


# Get image size
#0601258@N03  10424815813_e94629b1ec_o.jpg   
image = cv2.imread(single_image, 1)
# Height and width of the image
height, width = image.shape[:2]
# Number of channels
no_channels = 3
print("Image height {} width {}".format(height, width))
# Get the actual image path in a column.
labels_df['image_prefix'] = labels_df['face_id'].apply(lambda x : 'coarse_tilt_aligned_face.' + str(x) + ".")
labels_df['image_loc'] = image_path + labels_df['user_id'] + '/' +  labels_df['image_prefix'] + labels_df['original_image']

# Get the corresponding class integer from filter dict
labels_df['gender_class'] = labels_df['gender'].apply(lambda x: gen_filter[x])
labels_df['age_class']     = labels_df['age'].apply(lambda x: age_filter[x]) 


# Save the new labels files
labels_df[['user_id', 'original_image','face_id','age','gender','image_loc','age_class','gender_class']].to_csv(fnl_labels_path + 'labels.csv')

print("After Pre-processing")
print_stats(labels_df)

total_images = labels_df.shape[0] 

############### h5py file creation #####################

train_shape = (total_images, 256, 256, 3)

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
    resized_image = cv2.resize(image, (256, 256)) 
    hdf5_file["train_images"][i, ...] = resized_image[None]
    mean += resized_image / float(total_images)
    i+=1

hdf5_file["train_mean"][...] = mean
hdf5_file.close()

print("Finished creating hdf5 file.")



