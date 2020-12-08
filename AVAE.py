import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from utils import bilinear_kernel
MAX_LOGSTD  = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		self.basic_block = BasicBlock(3,3)


	def forward(self, x):
		x = self.basic_block(x)
		block1 = x.reshape(x.size(0), x.size(1), -1)
		



class AVAE(nn.Module):
	"""docstring for Generater"""
	### this is VAE 
	def __init__(self,latent_dim):
		super(AVAE, self).__init__()
		self.latent_dim = latent_dim
		self.encoder = Encoder(self.latent_dim).to(device)
		self.d_max = self.encoder.d_max
		self.decoder = Decoder(self.latent_dim,d_max = self.d_max).to(device)
		self.attn = Attention(input_pic)
		#self.kld_critique = nn.KLDivLoss()

	def forward(self,input_pic):
		input_pic = self.attn(input_pic)
		item_latent = self.encode(input_pic)
		reconstructed = self.decoder(item_latent)
		mse_loss,kld_loss = self.loss_function(reconstructed,input_pic,self.mu,self.var)
		vae_loss = mse_loss+kld_loss
		return item_latent,vae_loss
		
	def reparametrize(self,mu,var):
		self.std = torch.exp(0.5*var)
		eps = torch.randn_like(self.std)
		return mu + eps*self.std

	def encode(self,input_pic):
		self.mu,self.var = self.encoder(input_pic)
		self.var = torch.clamp(self.var,max = MAX_LOGSTD)
		z = self.reparametrize(self.mu,self.var)
		#z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
		return z
	def loss_function(self,reconstructed,input_pic,mu,var):
		
		#q = torch.empty(item_latent.size()).normal_(mean= 0 ,std = 1).to(device)
		recons_loss = F.mse_loss(reconstructed,input_pic)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
		#kld_loss = self.kld_critique(item_latent,q)
		sparse_loss = self.get_sparse_loss(self.mu)
		return recons_loss,kld_loss*8e-5#+sparse_loss*0.00001 上一次8e-4
	def get_sparse_loss(self,rho_hat,rho = 0.01):
		rho_hat = torch.mean(F.sigmoid(rho_hat), 1) 
		rho = torch.tensor([rho] * len(rho_hat)).to(device)
		loss = torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
		return loss

	def save(self):
		torch.save(self.encoder.state_dict(),"p1_encoder.pt")
		torch.save(self.decoder.state_dict(),"p1_decoder.pt")
	def load(self):
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.encoder.load_state_dict(
				torch.load("p1_encoder.pt"))
			self.decoder.load_state_dict(
				torch.load("p1_decoder.pt"))
		else:
			self.encoder.load_state_dict(torch.load(
				"p1_encoder.pt", map_location=lambda storage, loc: storage))
			self.decoder.load_state_dict(torch.load(
				"p1_decoder.pt",map_location=lambda storage,loc:storage))
		self.encoder.eval()
		self.decoder.eval()
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()
