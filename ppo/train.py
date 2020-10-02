import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from procgen import ProcgenEnv
from ppo.ppo import PPO

def train(params):

    torch.manual_seed(1)
    np.random.seed(1)


    env = ProcgenEnv(env_name="coinrun", render_mode="rgb_array")
    step = 0
    for i in range(100):
        
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1

    

    

    ppo = PPO(alpha=0.00001, in_dim=4, out_dim=2)

    ppo.policy_network.load_state_dict(torch.load("policy_params.pt"))

    ppo.train(env, n_epoch=1000, n_steps=800, render=False, verbos=False)


if __name__ == '__main__':
    
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', default=0.02, type=float)
    


    param = parser.parse_args()

    run(params)
    
    