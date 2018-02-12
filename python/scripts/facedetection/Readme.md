# Face Detection


There are three networks in total

1. 12Net.py
2. 24Net.py
3. 48Net.py



## Training Phase

### 12Net

Given an image find if a face is present in the image.
12Net.py

* data/facedetection/12net/facedetection/train/face
	* all face images from adience	
* data/facedetection/12net/facedetection/train/noface
	* all images from cifar-10

Read the images, resize them to 12 x 12 and train a binary classifier network.

