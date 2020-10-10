import torch
import numpy as np
from tqdm import tqdm

class Buffer(object):

	def __init__(self):

		self.log_probs = []
		self.k_log_probs = []
		self.states = []
		self.prev_states = []
		self.rewards = []
		self.disc_rewards = []
		self.advantages = []
		self.firsts = []

		self.mean_reward = []

	def store(self, state, reward, prev_state, first):
		self.store_state(state)
		self.store_reward(reward)
		self.store_firsts(first)
		self.store_prev_states(prev_state)

	def store_prev_states(self, prev_state):
		self.prev_states.append(prev_state)

	def store_state(self, state):
		self.states.append(state)

	def store_reward(self, reward):
		self.rewards.append(reward)

	def store_firsts(self, first):
		self.firsts.append(first)
		
	def store_log_probs(self, log_prob, k_log_prob):
		self.log_probs.append(log_prob)
		self.k_log_probs.append(k_log_prob)

	def store_advantage(self, advantage):
		self.advantages = advantage

	def clear(self):

		self.mean_reward.append(np.mean(self.disc_rewards).tolist())

		self.log_probs = []
		self.k_log_probs = []
		self.states = []
		self.prev_states = []
		self.rewards = []
		self.disc_rewards = []
		self.advantages = []
		self.firsts = []

	def get(self):

		states = torch.Tensor(self.states).reshape(-1, 3, 64, 64)
		prev_states = torch.Tensor(self.prev_states).reshape(-1, 3, 64, 64)
		log_probs = torch.cat(self.log_probs).reshape(-1, 1)
		k_log_probs = torch.cat(self.k_log_probs).reshape(-1, 1)
		disc_rewards = torch.Tensor(self.disc_rewards).reshape(-1, 1)
		#advantages = torch.Tensor(self.advantage)

		return states, log_probs, prev_states, k_log_probs, disc_rewards
