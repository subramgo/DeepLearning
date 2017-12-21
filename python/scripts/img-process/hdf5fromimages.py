import glob
import h5py
import sys 


image_path = sys.argv[1]
image_files = glob.glob(image_path, recursive = True)


hdf5_path = sys.argv[2]

train_shape = (total_images, 100, 100, 3)

# Open a hdf5 file and create earray
hdf5_file = h5py.File(hdf5_path, mode = 'w')

hdf5_file.create_dataset("test_images", train_shape, np.int8)






import os
import os.path
i = 0

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        path = os.path.join(dirpath, filename)


	if i % 1000 == 0 and i > 0:
    		print("Train data {}/{}".format(i, total_images))

	image = cv2.imread(path, cv2.IMREAD_COLOR)
	resized_image = cv2.resize(image, (100,100)) 
	hdf5_file["train_images"][i, ...] = resized_image[None]
	i+=1

hdf5_file.close()

print("Finished creating hdf5 file.")





