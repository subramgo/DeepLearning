"""
    Create a local CSV file with:
        * training/test split labels
        * class labels
    Labels are applied to the input data used by this model.
"""

import glob
import pandas as pd
import os
import cv2
import numpy as np
import h5py

from DLUtils import configs,datafeed

### Define filters for our model
age_filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}


### Read the image paths and labels
configs.load_configs('data.ini')
config = configs.get_section_dict()
labels_path = config['labels_path']
output_labels_path = config['output_labels_path']
image_path = config['image_path']

labels_files = glob.glob(labels_path)

df_from_each_file = (pd.read_table(f, sep='\t') for f in labels_files)
labels_df = pd.concat(df_from_each_file, ignore_index=True)



def print_stats(labels_df):
    """
    """
    print(labels_df.columns.values.tolist())
    print("Print total rows {}".format(labels_df.shape[0] ))
    print(labels_df.groupby(['age']).size())


print("Pre-processing Beginning")
print_stats(labels_df)

# Apply filters to get data needed for this model
labels_df = labels_df[labels_df['age'].isin(age_filter.keys())]

# Get the actual image path in a column.
labels_df['image_prefix'] = labels_df['face_id'].apply(lambda x : 'coarse_tilt_aligned_face.' + str(x) + ".")
labels_df['image_loc'] = image_path + labels_df['user_id'] + '/' +  labels_df['image_prefix'] + labels_df['original_image']

# Get the corresponding class integer from filter dict
labels_df['age_class'] = labels_df['age'].apply(lambda x: age_filter[x])

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

print("Pre-processing Complete")
print_stats(labels_df)
