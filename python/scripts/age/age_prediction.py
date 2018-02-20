"""
/***************************************************************

Age prediction model based on LAP (Looking at People) Challenge

http://chalearnlap.cvc.uab.es/

Training and evaluation data from Apparent Age v2 (CVPR 16)
http://chalearnlap.cvc.uab.es/dataset/19/description/


****************************************************************/
"""
import sys
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader 
import csv
from skimage import io, transform
from torchvision import transforms
from torch.autgrad import Variable
import torch.nn as nn
import torch.nn.functional as F



# Configure Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AgeDataSet(Dataset):
	""" Datset for Age Prediction """

	def __init__(self, labels_file, image_dir, transform=None):
		"""
		Args:

		"""
		self.image_dir = image_dir
		with open(labels_file, 'r') as f:
			reader = csv.reader(f)
			self.labels = list(reader)
		self.labels = self.labels[1:]
		#self.labels       = np.genfromtxt(labels_file, delimiter = ',')
		self.transform  = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sample = {}
		img_name = os.path.join(self.image_dir, self.labels[idx][0])

		image = io.imread(img_name)
		label  = float(self.labels[idx][1])


		sample['image'] = image
		sample['label']   = label

		if self.transform:
			sample = self.transform(sample)

		return sample




class Resize(object):
	""" Rescale the image to given dimension """

	def __init__(self, output_dim):
		assert isinstance(output_dim, tuple)
		self.output_dim = output_dim

	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		img = transform.resize(image, self.output_dim)
		return {"image": img, "label": label}

class ToTensor(object):
	""" Convert image to a tensor object """

	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image), 'label':label}

# Batch data dispenser
ageloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


class AgeNet(nn.Module):
	""" Neural Network to Predict the Age """

	def __init__(self):
		super(AgeNet, self).__init__()

	def forward(selx, x):
		





"""
if logger.getEffectiveLevel() == logging.DEBUG:

	resize = Resize((256, 256))
	tensor = ToTensor()

	
	dataset = AgeDataSet(labels_file='/Users/gsubramanian/Documents/Personel/Tech/blogs/Projects/DeepLearning/python/data/age/train_gt.csv'
		, image_dir='/Users/gsubramanian/Documents/Personel/Tech/blogs/Projects/DeepLearning/python/data/age/train/'
		, transform=transforms.Compose([resize, tensor]))

	print("Data set length {}".format(len(dataset)))
	for i in range(len(dataset)):
		sample = dataset[i]
		print(str(i), str(sample['image'].shape), str(sample['label']))
"""




