import torch

class Actor(torch.nn.Module):

	"""

	this is the policy network

	"""

	def __init__(self, hparams):

		super(Actor, self).__init__()

		self.l1 = torch.nn.Linear(1024, 256)
		self.l2 = torch.nn.Linear(256, 256)
		self.l3 = torch.nn.Linear(256, 1)

		self.relu = torch.nn.LeakyReLU()

		self.loss = torch.nn.MSELoss()


	def forward(self, x):

		out = x

		out = self.l1(out)
		out = self.relu(out)

		out = self.l2(out)
		out = self.relu(out)

		out = self.l3(out)
		
		return out