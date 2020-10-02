import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .buffer import Buffer

class PPO(object):

	def __init__(self, alpha=69.0005, in_dim=3, out_dim=2):
		# store parameters
		self.alpha = alpha
		self.input_dims = in_dim
		self.output_dims = out_dim

		# initialize policy network
		self.policy_network = PolicyNetwork(0.0005, in_dim, out_dim)

		# initialize old policy
		self.old_policy_network = PolicyNetwork(alpha=alpha, in_dim=in_dim, out_dim=out_dim)
		state_dict = self.policy_network.state_dict()
		self.old_policy_network.load_state_dict(state_dict)

		# initialize value network
		self.value_network = ValueNetwork(0.0001, in_dim, 1)

		# initialize vpg buffer
		self.buffer = Buffer()

		# historical episode length
		self.hist_length = []

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, len(s)).float()

		# get policy prob distrabution
		prediction = self.policy_network.forward(s)

		# get action probabilities
		action_probabilities = torch.distributions.Categorical(prediction)

		# sample action
		action = action_probabilities.sample()
		action = action.item()
		# get log prob
		log_prob = (action_probabilities.probs[0][action]).reshape(1, 1)

		# get old prob
		old_p = self.old_policy_network.forward(s)
		old_ap = torch.distributions.Categorical(old_p)
		old_log_prob = (old_ap.probs[0][action]).reshape(1, 1)

		return action, log_prob, old_log_prob

	def calculate_advantages(self, observation, prev_observation):

		observation = torch.from_numpy(observation).float()
		prev_observation = torch.from_numpy(prev_observation).float()

		# compute state value
		v = self.value_network.forward(prev_observation)

		# compute action function value
		q = self.value_network.forward(observation)

		# calculate advantage
		a = q - v + 1

		return a.detach().numpy()

	def update(self, iter=80):

		# returns buffer values as pytorch tensors
		observations, actions, old_actions, rewards, advantages = self.buffer.get_tensors()

		# set state dict
		state_dict = self.policy_network.state_dict()
		self.old_policy_network.load_state_dict(state_dict)

		r = (old_actions.detach())/actions

		# update policy
		self.policy_network.optimize(r, advantages, iter=1)

		# update value network
		self.value_network.optimize(observations, rewards, epochs=iter)

	def train(self, env, n_epoch, n_steps, render=False, verbos=True):

		# initialize step variable
		step = 0

		# historical episode length
		episode_lengths = [1]

		plt.ion()
		average_rewards = []
		highest_rewards = []

		# for n episodes or terminal state:
		for epoch in range(n_epoch):

			# initial reset of environment
			observation = env.reset()

			# store observation
			self.buffer.store_observation(observation)

			episode_lengths = [1]

			print("Epoch: {}".format(epoch))
			# for t steps:
			for t in range(n_steps):

				# increment step
				step += 1

				# render env screen
				if render: env.render()

				# get action, and network policy prediction
				action, log_prob, old_log_prob = self.act(observation)

				# store action
				self.buffer.store_action(log_prob)

				# store old action
				self.buffer.store_old_action(old_log_prob)

				# get state + reward
				observation, reward, done, info = env.step(action)

				# store observation
				self.buffer.store_observation(observation)

				# store rewards
				self.buffer.store_reward(reward)

				# calculate advantage
				a = self.calculate_advantages(self.buffer.observation_buffer[-1]
				, self.buffer.observation_buffer[-2])

				# store advantage
				self.buffer.store_advantage(a)

				# check if episode is terminal
				if done or t == n_steps-1:

					for s in reversed(range(1, step+1)):

						update = 0

						for k in reversed(range(1, s+1)):
							update += self.buffer.reward_buffer[-k]*(0.99**k)

						self.buffer.reward_buffer[-s] += update

					# change terminal reward to zero
					self.buffer.reward_buffer[-1] = 0

					# print time step
					if verbos:
						#print("Episode finished after {} timesteps".format(step+1))
						pass

					episode_lengths.append(step)

					# reset step counter
					step = 0

					# reset environment
					observation = env.reset()

			# update model
			self.update(iter=80)
			step=0
			self.buffer.clear_buffer()
			print("Average Episode Length: {}".format(
			np.sum(episode_lengths)/len(episode_lengths)))
			print("Largest Episode Length: {}".format(max(episode_lengths)))


			# plot
			average_rewards.append(np.sum(episode_lengths)/len(episode_lengths))
			highest_rewards.append(max(episode_lengths))
			plt.title("Reward per Epoch")
			plt.xlabel("Epoch")
			plt.ylabel("Reward")
			plt.plot(np.array(average_rewards), label="average reward")
			plt.plot(highest_rewards, label="highest reward")
			plt.legend(loc="upper left")
			plt.draw()
			"""
			if epoch%10 == 0:
				plt.savefig('reward_img/epoch{}.png'.format(epoch))
			"""
			plt.pause(0.0001)
			plt.clf()
			if average_rewards[-1] > 120:
				torch.save(self.policy_network.state_dict(), "policy_params.pt")

	def play(self, env):

		for i in range(1):

			# initial reset of environment
			observation = env.reset()
			done = False
			frame = 0
			while not done:
				frame+=1
				img = env.render(mode="rgb_array")
				scipy.misc.imsave('img/gif/img{}.jpg'.format(frame), img)

				# get action, and network policy prediction
				action, log_prob = self.act(observation)

				# get state + reward
				observation, reward, done, info = env.step(action)

def main():

	import gym

	torch.manual_seed(1)
	np.random.seed(1)

	env = gym.make('CartPole-v0')

	vpg = PPO(alpha=0.001, input_dims=4, output_dims=2)

	vpg.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)

if __name__ == "__main__":
	main()
