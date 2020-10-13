import torch
from tqdm import tqdm
import numpy as np

class Actor(torch.nn.Module):

	def __init__(self, actor_lr, epsilon):

		super(Actor, self).__init__()

		self.epsilon = epsilon
		self.define_network()
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=actor_lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)
		self.prev_params = self.parameters()

	def define_network(self):
		self.relu = torch.nn.ReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=0)
		self.l1 = torch.nn.Linear(1024, 512)
		self.l2 = torch.nn.Linear(512, 64)
		self.l3 = torch.nn.Linear(64, 15)
		self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)

	def loss(self, log_probs, k_log_probs, advantages):

		r_theta = torch.exp(log_probs-k_log_probs)

		clipped_r = torch.clamp(
			r_theta,
			1.0 - self.epsilon,
			1.0 + self.epsilon
			)

		return torch.mean(torch.min(r_theta*advantages, clipped_r*advantages))

	def forward(self, x):

		out = torch.Tensor(x).float().to(self.device)

		out = self.conv1(out)
		out = self.leaky_relu(out)
		out = self.conv2(out)
		out = self.leaky_relu(out)
		out = self.conv3(out)
		out = self.leaky_relu(out)
		out = self.conv4(out)
		out = self.leaky_relu(out)
		
		out = out.reshape(-1, 2*2*256)

		out = self.l1(out)
		out = self.leaky_relu(out)
		out = self.l2(out)
		out = self.leaky_relu(out)
		out = self.l3(out)

		out = self.relu(out)
		out = self.softmax(out)

		return out.to(torch.device('cpu:0'))

	def optimize(
		self,
		log_probs,
		k_log_probs,
		advantages,
		batch_sz=32
		):

		loss = self.loss(log_probs, k_log_probs, advantages)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		

def main():

	t1 = torch.ones(1, 3, 64, 64)
	pn = Actor(0.01, 3, 1)
	print(pn.device)
	print(pn(t1).shape)


if __name__ == "__main__":
	main()
