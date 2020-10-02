from ppo.ppo import PPO
import gym
import torch
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(1)
np.random.seed(1)


env = gym.make('CartPole-v0')

ppo = PPO(alpha=0.00001, in_dim=4, out_dim=2)

ppo.policy_network.load_state_dict(torch.load("policy_params.pt"))

ppo.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)

plt.plot(ppo.hist_length)
plt.show()
