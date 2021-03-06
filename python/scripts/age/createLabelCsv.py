import glob
import pandas as pd
import os
import cv2
import numpy as np
import h5py



# Labels folder
labels_path = '../data/Adience/*.txt'
fnl_labels_path = '../data/Adience/labels.csv'
image_path  = '../data/Adience/faces/'


# Filters
age_filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}
gen_filter = {'m':0,'f':1}

age_gender_filter = {
	'f-(0, 2)' :0,'m-(0, 2)' :1, 
           'f-(4, 6)':2 ,'m-(4, 6)':3 , 
           'f-(8, 12)':4, 'm-(8, 12)':5, 
           'f-(15, 20)':6,'m-(15, 20)':7,
           'f-(25, 32)':8,'m-(25, 32)':9,
           'f-(38, 43)':10, 'm-(38, 43)':11, 
           'f-(48, 53)':12, 'm-(48, 53)':13, 
           'f-(60, 100)':14,'m-(60, 100)':15
           }



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
# Get the actual image path in a column.
labels_df['image_prefix'] = labels_df['face_id'].apply(lambda x : 'coarse_tilt_aligned_face.' + str(x) + ".")
labels_df['image_loc'] = image_path + labels_df['user_id'] + '/' +  labels_df['image_prefix'] + labels_df['original_image']

# Get the corresponding class integer from filter dict
labels_df['gender_class'] = labels_df['gender'].apply(lambda x: gen_filter[x])
labels_df['age_class']     = labels_df['age'].apply(lambda x: age_filter[x]) 
labels_df['all_class']       = labels_df.apply(lambda row:  age_gender_filter[ row['gender'] + '-' + row['age'] ] , axis = 1)  

def get_type():
	eval_percentage = 0.2
	type_ = ''

	if np.random.rand() > eval_percentage:
		type_ = "train"
	else:
		type_ = "eval"
	return type_

labels_df['type'] = labels_df.apply(lambda row: get_type(), axis =1 )

# Save the new labels files
labels_df[['user_id', 'original_image','face_id','age','gender','image_loc','age_class','gender_class','all_class','type']].to_csv(fnl_labels_path)

print("After Pre-processing")
print_stats(labels_df)