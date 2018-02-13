import numpy as np
import sys
import cv2
from DLUtils.align_dlib import AlignDlib
import dlib
import scipy.misc

align_dlib = AlignDlib()
detector = dlib.get_frontal_face_detector()



uri = "rtsp://10.38.5.145/ufirststream/"
#uri = "rtsp://10.38.4.76/StreamId=1"
win = dlib.image_window()


def process_frame(image):
	#image = scipy.misc.toimage(frame)

	#image = cv2.imread(image)
	crop_dim = (96,96)
	#image = np.array(image, dtype='float32')
	#print(image.shape)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	dets, scores, idx = detector.run(image, 1)
	print(len(dets))
	aligned = None
	crops = []
	for i, d in enumerate(dets):
	 	print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
	 	crop = image[d.top():d.bottom(), d.left():d.right()]
	 	cv2.imwrite("cropped.jpg", crop)
	 	#if idx[0] == 0.0 or idx[0] == 3.0 or idx[0] == 4.0:
	 	crop = cv2.imread("cropped.jpg")
	 	aligned = align_dlib.align(crop_dim, crop)
	 	cv2.imwrite("aligned.jpg", aligned)
	 	aligned = cv2.imread("aligned.jpg")
	 	crops.append(aligned)
	
	if len(dets) > 0:
		win.clear_overlay()
		#image = scipy.misc.toimage(image)
		win.set_image(image)
		win.add_overlay(dets)
		dlib.hit_enter_to_continue()

		for c in crops:
			win.clear_overlay()
			#c = scipy.misc.toimage(c)

			win.set_image(c)
			dlib.hit_enter_to_continue()

	return crops



def read_opencv():
	cap = cv2.VideoCapture(uri)

	while(1):

	    ret, frame = cap.read()
	    if ret == False:
	    	print("Ret is false")
	    else:
	    	#image = scipy.misc.toimage(frame)
	    	image = frame
	    	#cv2.imshow('VIDEO', frame)
	    	#cv2.waitKey(0)
	    	aligned = process_frame(image)
	    	"""
	    	if len(aligned) > 0:
	    		for a in aligned:
	    			cv2.imshow('image', a)
	    			c = cv2.waitKey(0)
	    			if 'q' == char(c & 255):
	    				continue
	    	"""
	


	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	read_opencv()