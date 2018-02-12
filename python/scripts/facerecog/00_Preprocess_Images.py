import os  
import cv2 
import dlib
from DLUtils.align_dlib import AlignDlib
align_dlib = AlignDlib()
detector = dlib.get_frontal_face_detector()



register_path = '../data/facerecog/bocacafe/faces/'
output_path = '../data/facerecog/bocacafe/processedfaces/'

#register_path = '../data/facerecog/NamedFaces/'
#output_path = '../data/facerecog/ProcessedNamedFaces/'



def process_other_faces():
	for img in os.scandir(register_path):
		if not img.name.startswith('.') :
			crop_dim = (96,96)
			image = cv2.imread(img.path, )
			#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			dets, scores, idx = detector.run(image, 1, -1)
			for i, d in enumerate(dets):
				print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
			if len(dets) == 1:
			# Single Face Detected
				if idx[0] == 0.0:
				# No profile view
				# Front detected
					aligned = align_dlib.align(crop_dim, image)
					if aligned is not None:
						cv2.imwrite(output_path  + img.name, aligned)

def process_named_faces():

	for entry in os.scandir(register_path):
		if entry.is_dir() and entry.name != 'General':
			name = entry.name 
			print(name)
			for img in os.scandir(entry.path):
				if not img.name.startswith('.') :
					crop_dim = (96,96)
					image = cv2.imread(img.path, )
					#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					dets, scores, idx = detector.run(image, 1, -1)
					for i, d in enumerate(dets):
						print("Id {} Name {} Detection {}, score: {}, face_type:{}".format(i, name, d, scores[i], idx[i]))
					if len(dets) == 1:
					# Single Face Detected
						if idx[0] == 0.0 or idx[0] == 3.0 or idx[0] == 4.0:
						# No profile view
						# Front detected
							aligned = align_dlib.align(crop_dim, image)
							if aligned is not None:
								if not os.path.isdir(output_path + name):
									os.mkdir(output_path + name)
								cv2.imwrite(output_path + name + '/' + img.name, aligned)

if __name__ == "__main__":
	process_other_faces()
	#process_named_faces()
