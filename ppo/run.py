import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from procgen import ProcgenEnv
import gym3
from procgen import ProcgenGym3Env

#from ppo.ppo import PPO


def run():
    env = ProcgenGym3Env(num=1, env_name="coinrun", render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")
    step = 0
    for i in range(100):
        
        print(env.ac_space)
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1

    """    
    ppo.policy_network.load_state_dict(torch.load("policy_params.pt"))

    torch.manual_seed(1)
    np.random.seed(1)

    ppo = PPO(alpha=0.00001, in_dim=4, out_dim=2)

    ppo.policy_network.load_state_dict(torch.load("policy_params.pt"))

    ppo.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)

    plt.plot(ppo.hist_length)
    plt.show()
    """


if __name__ == '__main__':
    
    run()
    
    