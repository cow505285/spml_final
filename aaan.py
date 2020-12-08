import torch.nn as nn
import torch.nn.functional as F
import torch
from vae import VAE
from collections import namedtuple
class Encoder(nn.Module):
	def __init__(self,latent_dim,input_channel = 3):
		super(Encoder,self).__init__()
		self.latent_dim = latent_dim
		self.layer_count = 4

		inputs = input_channel
		mul = 1
		out_dim = 36
		for i in range(self.layer_count):
			setattr(self,"encode_conv%d"%(i+1),nn.Conv2d(inputs,out_dim*mul,kernel_size = 4, stride = 2,padding = 1))
			setattr(self,"encode_bnorm%d"%(i+1),nn.BatchNorm2d(out_dim*mul))
			inputs = out_dim*mul
			mul *= 2
		self.d_max = inputs

		
		self.get_mu = nn.Linear(inputs*4*4,self.latent_dim)
		self.get_var = nn.Linear(inputs*4*4,self.latent_dim)
	def forward(self,input_pic):
		x = input_pic
		for i in range(self.layer_count):
			x = getattr(self,"encode_conv%d"%(i+1))(x)
			x = F.leaky_relu(getattr(self,"encode_bnorm%d"%(i+1))(x))
		result = torch.flatten(x,start_dim = 1)

		mu = self.get_mu(result)
		var = self.get_var(result)
		var = F.log_softmax(var,dim = -1)
		return mu,var

class Decoder(nn.Module):
	def __init__(self,latent_dim,d_max):
		super(Decoder,self).__init__()
		self.latent_dim = latent_dim
		self.layer_count = 4
		self.d_max = d_max
		inputs = d_max
		out_dim = 36
		mul = d_max//out_dim//2
		for i in range(self.layer_count-1):
			setattr(self,"decode_conv%d"%(i+1),nn.ConvTranspose2d(inputs,out_dim*mul,kernel_size = 4, stride = 2,padding = 1))
			setattr(self,"decode_bnorm%d"%(i+1),nn.BatchNorm2d(out_dim*mul))
			inputs = out_dim*mul
			mul //= 2
		setattr(self,"decode_conv%d"%(self.layer_count),nn.ConvTranspose2d(inputs,3,kernel_size = 4 , stride = 2,padding = 1))
		setattr(self,"decode_bnorm%d"%(self.layer_count),nn.BatchNorm2d(3))
		
		self.decode_input = nn.Linear(self.latent_dim,d_max*4*4)

	def forward(self,latent):
		x = self.decode_input(latent)
		x = x.view(-1,self.d_max,4,4)
		x = F.leaky_relu(self.decode_conv1(x))
		#x = self.bdnorm1(x)
		x = F.leaky_relu(self.decode_conv2(x))
		#x = self.bdnorm2(x)
		x = F.leaky_relu(self.decode_conv3(x))
		#x = self.bdnorm3(x)
		x = self.decode_conv4(x)
		#x = self.bdnorm4(x)
		reconstructed =  x.tanh()
		
		return reconstructed

class Feature_extractor(nn.Module):
	"""docstring for Feature_extractor"""
	def __init__(self):
		super(Feature_extractor, self).__init__()
		self.layer_count = 2
		self.maxpool = nn.MaxPool2d(2)
		self.dropout = nn.Dropout2d(p = 0.3)
		inputs = 3
		dim = 64
		mul = 1
		for i in range(self.layer_count):
			setattr(self,"conv%d"%(i+1),nn.Conv2d(inputs,dim*mul,kernel_size = 5))
			setattr(self,"bnorm%d"%(i+1),nn.BatchNorm2d(dim*mul))
			inputs = dim*mul
			mul*=2
		self.d_max = inputs

	def forward(self,pic):
		x = pic
		for i in range(self.layer_count):
			x = F.leaky_relu(getattr(self,"maxpool")(getattr(self,"bnorm%d"%(i+1))(getattr(self,"conv%d"%(i+1))(x))))
			x = self.dropout(x)
		result = torch.flatten(x,start_dim = 1)
		return result

class BasicBlock(nn.Module):
	def __init__(self,in_channel,out_channel):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channel,out_channel, kernel_size=3,stride=1,padding=1, groups=1, bias=False, dilation=1)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(out_channel,out_channel, kernel_size=3,stride=1,padding=1, groups=1, bias=False, dilation=1)
		self.bn2 = nn.BatchNorm2d(out_channel)

	def forward(self, x) :
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += identity
		out = self.relu(out)
		print("out.size()",out.size())

		return out
class Generator(object):
	"""docstring for Generator"""
	def __init__(self, latent_dim):
		super(Generator, self).__init__()
		self.latent_dim =  latent_dim
		self.encoder = Encoder(self.latent_dim,input_channel = 3)
		self.d_max = self.encoder.d_max
		self.decoder = Decoder(self.latent_dim,d_max = self.d_max)
	def forward(self,input):
		bottle_nack = self.encoder(input)
		reconstructed = self.decoder(bottle_nack)
		return reconstructed
		
class Discriminator(object):
	"""docstring for ClassName"""
	def __init__(self):
		super(Discriminator, self).__init__()
		self.feature_extractor = Feature_extractor()
		self.d_max = self.feature_extractor.d_max
		self.classifier = nn.Sequential(nn.Linear(2*self.d_max*4*4,200),nn.BatchNorm1d(200),nn.ReLU(),nn.Linear(200,1))
	def forward(self,input_x,input_y):
		feat_x = self.feature_extractor(input_x)
		feat_y = self.feature_extractor(input_y)
		feat_cat = torch.cat((feat_x,feat_y),dim = 1)
		pred = self.classifier(feat_cat)
		return pred

class ADDN(object):
	"""docstring for ClassName"""
	def __init__(self,latent_dim = 64):
		super(ADDN, self).__init__()
		self.latent_dim = latent_dim
		self.vae = VAE(self.latent_dim)
		self.basic_block = BasicBlock(3,64)
		self.G1 = Generator(self.latent_dim)
		self.G2 = Generator(self.latent_dim)
		self.Identity_Discriminator = Discriminator()
		self.Sample_Discriminator = Discriminator()
	def forward(self,origin_face,target_face):
		results = namedtuple('target_latent')
		target_latent,vae_loss = self.vae(target_face)

		

