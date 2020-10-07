import torch
from tqdm import tqdm
import numpy as np


from .actor import Actor
from .critic import Critic
from .buffer import Buffer

class PPO(object):

	def __init__(self, alpha=0.0005, in_dim=3, out_dim=2):
		# store parameters
		self.alpha = alpha
		self.input_dims = in_dim
		self.output_dims = out_dim

		# initialize policy network
		self.actor = PolicyNetwork(0.0005, in_dim, out_dim)

		# initialize old policy
		self.old_actor = PolicyNetwork(alpha=alpha, in_dim=in_dim, out_dim=out_dim)
		state_dict = self.actor.state_dict()
		self.old_actor.load_state_dict(state_dict)

		# initialize value network
		self.critic = ValueNetwork(0.0001, in_dim, 1)

		# initialize vpg buffer
		self.buffer = Buffer()

		# historical episode length
		self.hist_length = []

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, len(s)).float()

		# get policy prob distrabution
		prediction = self.actor.forward(s)

		# get action probabilities
		action_probabilities = torch.distributions.Categorical(prediction)

		# sample action
		action = action_probabilities.sample()
		action = action.item()
		# get log prob
		log_prob = (action_probabilities.probs[0][action]).reshape(1, 1)

		# get old prob
		old_p = self.old_actor.forward(s)
		old_ap = torch.distributions.Categorical(old_p)
		old_log_prob = (old_ap.probs[0][action]).reshape(1, 1)

		return action, log_prob, old_log_prob

	def calculate_advantages(self, observation, prev_observation):

		observation = torch.from_numpy(observation).float()
		prev_observation = torch.from_numpy(prev_observation).float()

		# compute state value
		v = self.critic.forward(prev_observation)

		# compute action function value
		q = self.critic.forward(observation)

		# calculate advantage
		a = q - v + 1

		return a.detach().numpy()

	def update(self, iter=80):

		# returns buffer values as pytorch tensors
		observations, actions, old_actions, rewards, advantages = self.buffer.get_tensors()

		# set state dict
		state_dict = self.actor.state_dict()
		self.old_actor.load_state_dict(state_dict)

		r = (old_actions.detach())/actions

		# update policy
		self.actor.optimize(r, advantages, iter=1)

		# update value network
		self.critic.optimize(observations, rewards, epochs=iter)



def main():

	import gym

	torch.manual_seed(1)
	np.random.seed(1)

	env = gym.make('CartPole-v0')

	ppo = PPO(alpha=0.001, input_dims=4, output_dims=2)

	train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)

if __name__ == "__main__":
	main()
