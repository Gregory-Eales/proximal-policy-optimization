import torch
from tqdm import tqdm
import numpy as np
import random
import gym3


from .actor import Actor
from .critic import Critic
from .buffer import Buffer


class RandomAgent():

	def __init__(self, n_envs):

		self.n_envs = n_envs
		self.reward = []

	def act(self, state):
		return np.random.randint(0, high=15, size=[self.n_envs, ])

	def store(self, action, state, reward, prev_state):
		self.reward.append(reward)

	def update(self):
		pass


class PPO(object):

	def __init__(
			self,
			actor_lr,
			critic_lr,
			batch_sz,
			gamma,
			epsilon,
			actor_epochs,
			critic_epochs,
	):

		self.actor_lr = actor_lr
		self.critic_lr  = critic_lr
		self.batch_sz = batch_sz
		self.gamma = gamma
		self.epsilon = epsilon
		self.actor_epochs = actor_epochs
		self.critic_epochs = critic_epochs

		# initialize policy network
		self.actor = Actor(
			actor_lr=actor_lr,
			actor_epochs=actor_epochs,
			epsilon=epsilon
		)

		self.k_actor = Actor(
			actor_lr=actor_lr,
			actor_epochs=actor_epochs,
			epsilon=epsilon
		)

		self.critic = Critic(
			critic_lr=critic_lr,
			critic_epochs=critic_epochs
		)

		self.transfer_weights()

		self.buffer = Buffer()

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

		# get policy prob distrabution
		prediction = self.actor(s)
		action_probabilities = torch.distributions.Categorical(prediction)
		actions = action_probabilities.sample()

		log_prob = action_probabilities.log_prob(actions)

		k_p = self.k_actor(s).detach()
		k_ap = torch.distributions.Categorical(k_p)
		k_log_prob = k_ap.log_prob(actions)

		self.buffer.store_log_probs(log_prob, k_log_prob)

		return actions.detach().numpy()

	def discount_rewards(self):
		firsts = np.array(self.buffer.firsts).reshape([-1, 1]).astype('int32')
		rewards = np.array(self.buffer.rewards).reshape([-1, 1])

		for i in reversed(range(rewards.shape[0]-1)):
			rewards[i] += rewards[i+1]*self.gamma*(1-firsts[i])

		self.buffer.disc_rewards=rewards

	def transfer_weights(self):
		state_dict = self.actor.state_dict()
		self.k_actor.load_state_dict(state_dict)

	def store(self, state, reward, prev_state, first):
		self.buffer.store(state, reward, prev_state, first)

	def calculate_advantages(self, states, prev_states):

		state = torch.from_numpy(states).float()
		prev_state = torch.from_numpy(prev_states).float()

		# compute state value
		v = self.critic(prev_state)
		q = self.critic(state)
		a = q - v + 1

		return a.detach().numpy()

	def update(self, iter=80):

		# returns buffer values as pytorch tensors
		s, lp, p_s, k_lp, d_r = self.buffer.get_tensors()

		self.transfer_weights()

		adv = self.calculate_advantages(s, p_s)
		
		self.actor.optimize(
			log_probs=lp,
			k_log_probs=k_lp,
			advantages=adv
			)

		self.critic.optimize(
			states=s,
			rewards=d_r,
			epochs=self.critic_epochs,
			batch_sz=self.batch_size
			)

	def get_rewards(self):
		return self.buffer.rewards


def main():

	import gym

	torch.manual_seed(1)
	np.random.seed(1)

	env = gym.make('CartPole-v0')

	ppo = PPO(alpha=0.001, input_dims=4, output_dims=2)

	train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)


if __name__ == "__main__":
	main()
