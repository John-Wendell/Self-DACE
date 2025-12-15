import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):




	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

		self.transform = transforms.Compose([

			transforms.RandomCrop((512, 512)),
   
			# transforms.RandomCrop((256, 256)),

			transforms.ToTensor(),


		])
  
		self.aug = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomRotation(90),
		])
		

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		#
		# data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS)

		# data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = self.transform(data_lowlight)
		data_lowlight = self.aug(data_lowlight)
  
		random_flag = np.random.uniform(0.,1.)
		if random_flag > 0.5:
				random_gamma = np.random.uniform(1, 2)
				data_lowlight = data_lowlight ** (random_gamma)
		# data_lowlight = torch.from_numpy(data_lowlight).float()
		# data_lowlight = data_lowlight.permute(2,0,1)
		

		# data_lowlight_distorted = data_lowlight.clone()
		# get size of data_lowlight
		# _, h, w = data_lowlight.size()
		# ### random mask 1/10 pixels of data_lowlight
		# mask = torch.ones_like(data_lowlight)
		# mask = mask.view(-1, h*w)
		# mask[:, torch.randint(h*w, (1, int(h*w/5)))[0]] = 0
		# mask = mask.view(-1, h, w)
		# data_lowlight_masked = data_lowlight * mask


		return {'data_lowlight': data_lowlight, 'path': data_lowlight_path}

	def __len__(self):
		return len(self.data_list)

