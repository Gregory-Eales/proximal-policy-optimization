import torch
import numpy as np
from tqdm import tqdm


class Critic(torch.nn.Module):

	def __init__(self, critic_lr, critic_epochs):


		# inherit from nn module class
		super(Critic, self).__init__()

		# initialize_network
		self.initialize_network()

		# define optimizer
		self.optimizer = torch.optim.Adam(
			lr=critic_lr,
			params=self.parameters()
			)

		# define loss
		self.loss = torch.nn.MSELoss()

		# get device
		self.device = torch.device(
			'cuda:0' if torch.cuda.is_available() else 'cpu:0')
		
		self.to(self.device)

	# initialize network

	def initialize_network(self):

				# define network components
		self.fc1 = torch.nn.Linear(1024, 64)
		self.fc2 = torch.nn.Linear(64, 64)
		self.fc3 = torch.nn.Linear(64, 1)
		self.relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)

	def forward(self, x):

		out = torch.Tensor(x).to(self.device)

		out = self.conv1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.relu(out)
		out = self.conv4(out)
		out = self.relu(out)

		out = out.reshape(-1, 1024)

		out = self.fc1(out)
		out = self.tanh(out)
		out = self.fc2(out)
		out = self.tanh(out)
		out = self.fc3(out)
		out = self.relu(out)

		return out.to(torch.device('cpu:0'))

	def optimize(
		self,
		states,
		rewards,
		epochs,
		batch_sz
		):

		n_samples = rewards.shape[0]
		num_batch = int(n_samples//batch_sz)

		for i in tqdm(range(epochs)):

			for b in range(num_batch):

				s = states[b*batch_sz:(b+1)*batch_sz]
				r = rewards[b*batch_sz:(b+1)*batch_sz]
			   
				p = self.forward(s)
				loss = self.loss(p, r)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			s = states[(num_batch)*batch_sz:]
			r = rewards[(num_batch)*batch_sz:]

			p = self.forward(s)
			loss = self.loss(p, r)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

def main():
	pass

if __name__ == "__main__":
	main()
