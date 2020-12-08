import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import models, transforms as T
from PIL import Image
from os import listdir
from argparse import ArgumentParser
from aaan import Aaan
from torchvision.utils import save_image
device = torch.device("cuda")
random.seed(0)
def init_args():
	parser = ArgumentParser()
	parser.add_argument('--batch_size',default = 64,type=int)
	parser.add_argument('--epoch_num',default = 32,type=int)
	parser.add_argument('--input_size',default = 64,type =int)
	parser.add_argument('--root_dir',default ='cfp-dataset/Data/Images')
	parser.add_argument('--text_file',default = 'cfp-dataset/Data/list_name.txt')
	return parser.parse_args()
class myImageDataset(Dataset):
	"""Face Landmarks dataset."""
	def __init__(self, root_dir, transform=None, classes = None):
		self.root_dir = root_dir
		self.transform = transform
		self.classes = classes
		
	def __len__(self):
		return 7000

	def __getitem__(self, index):
		pic_class = index//500 + 1
		pic_class = '%03d'% pic_class
		pic_idx =  int(index%14) + 1
		if pic_idx < 11:    
			pid_idx = '%02d'% pic_idx
			pic_name = f'/{pic_class}/frontal/{pid_idx}.jpg'
		else:
			pic_idx -= 10
			pid_idx = '%02d'% pic_idx
			pic_name = f'/{pic_class}/profile/{pid_idx}.jpg'
		img_name = self.root_dir + pic_name
		img = Image.open(img_name).convert('RGB')
		ori_img = np.array(img, dtype='int')
		if self.transform is not None:
			trans_img = self.transform(img)
		label = torch.tensor(int(index//500))
		return trans_img, label, self.classes[int(pic_class) - 1]



def main():
	args = init_args()
	classes = []
	with open (args.text_file, 'r') as f:
		for i in f.read().split('\n'):
			classes.append(i)

	mean, std = [0.5], [0.5]
	transform = T.Compose([
			T.Resize(args.input_size),
			T.ToTensor(),
			T.Normalize(mean,std)
			])
	dataset = myImageDataset(root_dir, transform, classes )

	trainloader = DataLoader(dataset, batch_size = args.batch_size, drop_last = True, shuffle= True, worker_init_fn=0)


	model = aaan()

	return
if __name__ == '__main__':
	main()