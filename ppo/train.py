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


def train(agent, env, n_epoch, n_steps):

	for epoch in tqdm(range(n_epoch)):

		reward, prev_state, prev_first = env.observe()

		for i in range(n_steps):

			action = agent.act(prev_state['rgb'])

			env.act(action)

			reward, state, first = env.observe()

			agent.store(state['rgb'], reward, prev_state['rgb'], prev_first)

			prev_state = state
			prev_first = first

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
	actor_epochs,
	critic_epochs,
):

	exp_path = create_exp_dir(experiment_name)

	agent = PPO(
		actor_lr=actor_lr,
		critic_lr=critic_lr,
		batch_sz=batch_sz,
		gamma=gamma,
		epsilon=epsilon,
		actor_epochs=actor_epochs,
		critic_epochs=critic_epochs,
	)

	#agent = RandomAgent(n_envs=n_envs)

	env = ProcgenGym3Env(num=n_envs, env_name="coinrun")
	train(agent, env, n_episodes, n_steps)
	generate_graphs(agent, exp_path)

	print("##########")
	print("Firsts: ", np.array(agent.buffer.firsts).shape)
	print("Rewards: ", np.array(agent.buffer.rewards).shape)
	print("States: ", np.stack(agent.buffer.states).shape)
	print("##########")
	print(np.array(agent.buffer.firsts).reshape([-1, 1]).shape)
	print(np.array(agent.buffer.rewards).reshape([-1, 1]).shape)
	print(np.concatenate(agent.buffer.states).shape)
	print(np.array(agent.buffer.firsts).reshape([-1, 1]).astype('int32'))
	print("##########")

	agent.discount_rewards()
	plt.show()

	plt.plot(agent.buffer.disc_rewards)
	plt.show()


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
	parser.add_argument('--n_episodes', default=10, type=int)
	parser.add_argument('--n_steps', default=1000, type=int)
	parser.add_argument('--batch_sz', default=16, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--actor_epochs', default=10, type=int)
	parser.add_argument('--critic_epochs', default=10, type=int)
	parser.add_argument('--n_envs', default=3, type=int)

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
		actor_epochs=params.actor_epochs,
		critic_epochs=params.critic_epochs,
	)
