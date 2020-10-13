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
			critic_epochs,
	):

		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.batch_sz = batch_sz
		self.gamma = gamma
		self.epsilon = epsilon
		self.critic_epochs = critic_epochs

		# initialize policy network
		self.actor = Actor(
			actor_lr=actor_lr,
			epsilon=epsilon
		)

		self.k_actor = Actor(
			actor_lr=actor_lr,
			epsilon=epsilon
		)

		self.critic = Critic(
			critic_lr=critic_lr,
			critic_epochs=critic_epochs
		)

		self.transfer_weights()

		self.buffer = Buffer()

	def act(self, s):

		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

		prediction = self.actor.forward(s)
		action_probabilities = torch.distributions.Categorical(prediction)
		actions = action_probabilities.sample()
		log_prob = action_probabilities.log_prob(actions)

		k_p = self.k_actor(s)
		k_ap = torch.distributions.Categorical(k_p)
		k_log_prob = k_ap.log_prob(actions.detach())

		self.buffer.store_log_probs(log_prob, k_log_prob.detach())

		return actions.detach().numpy()

	def discount_rewards(self):
		firsts = np.array(self.buffer.firsts).reshape([-1, 1]).astype('int32')
		rewards = np.array(self.buffer.rewards).reshape([-1, 1])
		for i in tqdm(reversed(range(rewards.shape[0]-1))):
			rewards[i] += rewards[i+1]*self.gamma*(1-firsts[i])

		self.buffer.disc_rewards = rewards

	def transfer_weights(self):
		state_dict = self.actor.state_dict()
		self.k_actor.load_state_dict(state_dict)

	def store(self, state, reward, prev_state, first):
		self.buffer.store(state, reward, prev_state, first)

	def calculate_advantages(self, states, prev_states, batch_sz=64):

		a = []

		print(states.shape)
		n_samples = states.shape[0]
		num_batch = int(n_samples//batch_sz)

		for b in tqdm(range(num_batch)):

			s = states[b*batch_sz:(b+1)*batch_sz]
			p_s = prev_states[b*batch_sz:(b+1)*batch_sz]
			v = self.critic(p_s).detach()
			q = self.critic(s).detach()
			a.append(q - v + 1)
			

		s = states[(num_batch)*batch_sz:]
		p_s = prev_states[(num_batch)*batch_sz:]
		v = self.critic(p_s).detach()
		q = self.critic(s).detach()
		a.append(q - v + 1)

		a = torch.cat(a)

		return a

	def update(self):

		self.discount_rewards()
	
		# returns buffer values as pytorch tensors
		s, lp, p_s, k_lp, d_r = self.buffer.get()

		adv = self.calculate_advantages(s, p_s)

		self.transfer_weights()

		self.critic.optimize(
			states=s,
			rewards=d_r,
			epochs=self.critic_epochs,
			batch_sz=self.batch_sz
			)

		self.actor.optimize(
			log_probs=lp,
			k_log_probs=k_lp,
			advantages=adv
			)

		self.buffer.clear()

	def get_rewards(self):
		return self.buffer.mean_reward

def main():
	pass

if __name__ == "__main__":
	main()
