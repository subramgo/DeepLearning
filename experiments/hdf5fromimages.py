import glob
import h5py
import sys
import numpy as np
import cv2
import os
import os.path


image_path = sys.argv[1]
hdf5_path = sys.argv[2]


# Open a hdf5 file and create earray
hdf5_file = h5py.File(hdf5_path, mode = 'w')



# Do a loop to find the number of files
i = 0

for dirpath, dirnames, filenames in os.walk(image_path):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        i+=1

print(i)
hdf5_file.create_dataset("test_images", (i,100,100,3), np.int8)









i = 0

for dirpath, dirnames, filenames in os.walk(image_path):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        if i % 1000 == 0 and i > 0:
                print("Train data {}".format(i))
        print(os.path.join(dirpath,filename))
        image = cv2.imread(os.path.join(dirpath,filename), cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (100,100))
        hdf5_file["test_images"][i, ...] = resized_image[None]
        i+=1

hdf5_file.close()
print("Finished creating hdf5 file.")