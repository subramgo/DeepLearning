import os
import numpy as np 


csv_file_path = '../data/facedetection/final.csv'
face_path = '../data/Adience/faces'
nonface_path = '../data/cifar-10/train/'



eval_percentage = 0.2

f = open(csv_file_path, 'w')
f.write("type,class,directory\n")

type_ = ""
class_ = "1" # face
directory_= ""
for subdir, dirs, files in os.walk(face_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
	        directory_ = os.path.join(subdir, file)
	        if np.random.rand() > eval_percentage:
	        	type_ = "train"
	        else:
	        	type_ = "eval"
	        f.write(type_+ "," + class_ + "," + directory_ + "\n")

class_ = "0" # No face
for subdir, dirs, files in os.walk(nonface_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
	        directory_ = os.path.join(subdir, file)
	        if np.random.rand() > eval_percentage:
	        	type_ = "train"
	        else:
	        	type_ = "eval"
	        f.write(type_+ "," + class_ + "," + directory_ + "\n")

f.close()





