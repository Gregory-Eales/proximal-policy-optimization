from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import random
import gym3
from procgen import ProcgenGym3Env
import time
from matplotlib import pyplot as plt

from modules import *
from utils import *

torch.autograd.set_detect_anomaly(True)


def train(agent, env, n_epoch, n_steps):

	counter = 0
	ran = False

	for epoch in tqdm(range(n_steps)):

		reward, prev_state, prev_first = env.observe()

		action = agent.act(prev_state['rgb'])

		env.act(action)

		reward, state, first = env.observe()

		agent.store(state['rgb'], reward, prev_state['rgb'], prev_first)
		prev_state = state
		prev_first = first

		if first:
			ran = False
			counter+=1

		if counter % 10 == 0 and counter != 0 and not ran:
			agent.update()
			ran = True


def train_multi(agent, env, n_epoch, n_steps):

	for epoch in range(n_epoch):

		reward, prev_state, prev_first = env.observe()

		for i in tqdm(range(n_steps)):

			action = agent.act(prev_state['rgb'])

			env.act(action)

			reward, state, first = env.observe()
			prev_state = state
			prev_first = first


def train_single(agent, env, n_epoch, n_steps):

	prev_state = env.reset()

	#for epoch in tqdm(range(n_epoch)):

	for i in tqdm(range(n_steps)):

		action = agent.act(prev_state)

		state, reward, done, info = env.step(action[0])

		if done:
			pass

		prev_state = state
			
		#agent.update()


def run_experiment(
	experiment_name,
	environment_name,
	log,
	graph,
	random_seeds,
	n_episodes,
	n_steps,
	n_envs,
	epsilon,
	batch_sz,
	critic_lr,
	actor_lr,
	gamma,
	critic_epochs,
):

	exp_path = create_exp_dir(experiment_name)

	agent = PPO(
		actor_lr=actor_lr,
		critic_lr=critic_lr,
		batch_sz=batch_sz,
		gamma=gamma,
		epsilon=epsilon,
		critic_epochs=critic_epochs,
	)

	# agent = RandomAgent(n_envs=n_envs)


	env = ProcgenGym3Env(
			num=n_envs,
			env_name="coinrun",
			render_mode="rgb_array",
			center_agent=False,
			num_levels=1,
			start_level=2,
			)

	train(agent, env, n_episodes, n_steps)
	generate_graphs(agent, exp_path)

	print(len(agent.buffer.mean_reward))
	print(np.array(agent.buffer.mean_reward).shape)
	print(np.stack(agent.buffer.mean_reward).shape)
	print(agent.buffer.mean_reward)


	plt.plot(agent.buffer.mean_reward)
	plt.show()

	"""
	import gym
	env = gym.make("procgen:procgen-coinrun-v0")
	env = ProcgenGym3Env(num=n_envs, env_name="coinrun")
	train_single(agent, env, n_episodes, n_steps)
	"""

if __name__ == '__main__':

	parser = ArgumentParser(add_help=False)

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument('--environment_name', default="couinrun")

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training params
	parser.add_argument('--random_seeds', default=list(range(10)), type=list)
	parser.add_argument('--n_episodes', default=20, type=int)
	parser.add_argument('--n_steps', default=100000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--critic_epochs', default=20, type=int)
	parser.add_argument('--n_envs', default=1, type=int)

	# model params
	parser.add_argument('--actor_lr', default=2e-1, type=float)
	parser.add_argument('--critic_lr', default=2e-1, type=float)
	parser.add_argument('--epsilon', default=0.3, type=float)

	params = parser.parse_args()

	run_experiment(
		experiment_name=params.experiment_name,
		environment_name=params.environment_name,
		log=params.log,
		graph=params.graph,
		random_seeds=params.random_seeds,
		n_episodes=params.n_episodes,
		n_steps=params.n_steps,
		n_envs=params.n_envs,
		epsilon=params.epsilon,
		batch_sz=params.batch_sz,
		critic_lr=params.critic_lr,
		actor_lr=params.actor_lr,
		gamma=params.gamma,
		critic_epochs=params.critic_epochs,
	)
