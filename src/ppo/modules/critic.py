import torch



class Critic(torch.nn.Module):

	"""
	critic or value network used to make value predictions from states

	"""

	def __init__(self, hparams):

		super(Critic, self).__init__()

		self.l1 = torch.nn.Linear(1024, 256)
		self.l2 = torch.nn.Linear(256, 256)
		self.l3 = torch.nn.Linear(256, 15)

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