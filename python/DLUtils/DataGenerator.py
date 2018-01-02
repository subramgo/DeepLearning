import numpy as np
import h5py
from keras.utils import np_utils


def adience_train_generator(batchsize):
	filepath = '../data/Adience/hdf5/adience-100.h5'
	dimensions = (batchsize, 100,100 ,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['train_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['train_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['train_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize

			# Shuffle
			p = np.random.permutation(len(y_train_onecoding))
			yield (x_train[p], y_train_onecoding[p])
		f.close()

def adience_eval_generator(batchsize):
	filepath = '../data/Adience/hdf5/adience-100.h5'
	dimensions = (batchsize, 100,100 ,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['eval_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['eval_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['eval_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize

			# Shuffle
			p = np.random.permutation(len(y_train_onecoding))
			yield (x_train[p], y_train_onecoding[p])
		f.close()

def vgg_train_generator(batchsize):
	filepath = '../data/facedetection/vgg.h5'

	dimensions = (batchsize, 200,200 ,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['train_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['train_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['train_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()

def vgg_eval_generator(batchsize):
	filepath = '../data/facedetection/vgg.h5'

	dimensions = (batchsize, 200, 200,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['eval_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['eval_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['eval_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()


def face_12net_train_generator(batchsize):
	filepath = '../data/facedetection/12net.h5'

	dimensions = (batchsize, 12,12 ,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['train_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['train_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['train_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()

def face_12net_eval_generator(batchsize):
	filepath = '../data/facedetection/12net.h5'

	dimensions = (batchsize, 12, 12,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['eval_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['eval_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['eval_labels'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 2)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()


def adience_datagenerator(filepath, batchsize):
	dimensions = (batchsize, 100, 100,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['train_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['train_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['train_labels'][n_entries:n_entries+batchsize]
			y_train_1 = y_train[:,0]
			y_train_2 = y_train[:,1]
			y_train_1_onecoding = np_utils.to_categorical(y_train_1, 8)
			y_train_2_onecoding = np_utils.to_categorical(y_train_2, 2)
			y_train_onecoding = np.concatenate((y_train_2_onecoding, y_train_1_onecoding), axis = 1)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()

def adience_datagenerator_16classes(filepath, batchsize):
	""" Extract the 16 dimensional Y variable"""
	dimensions = (batchsize, 100, 100,3)
	while 1:

		f = h5py.File(filepath, "r")
		filesize = len(f['train_labels'])

		n_entries = 0
		while n_entries < (filesize - batchsize):
			x_train= f['train_images'][n_entries : n_entries + batchsize]
			x_train= np.reshape(x_train, dimensions).astype('float32')

			y_train = f['train_single_label'][n_entries:n_entries+batchsize]
			y_train_onecoding = np_utils.to_categorical(y_train, 16)

			n_entries += batchsize
			yield (x_train, y_train_onecoding)
		f.close()

if __name__ == '__main__':
	hdf5_path = '../data/Adience/hdf5/adience-100.h5'
	vals = adience_datagenerator(hdf5_path, 4)
	x_train, y_train = next(vals)
	print(x_train.shape)

	vals1 = adience_datagenerator_16classes(hdf5_path, 4)
	x_train,y_train = next(vals1)
	print(x_train.shape)
	print(y_train.shape)
	print(y_train)

