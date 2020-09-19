import torch

from .encoder import Encoder
from .decoder import Decoder



class AutoEncoder(torch.nn.Module):

	"""
	this is an auto encoder used to learn latent representations of the data
	"""

	def __init__(self):
		

		self.encoder = Encoder()
		self.decoder = Decoder()


