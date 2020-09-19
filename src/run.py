"""
This file defines the core research contribution   
"""
import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import gym3
from procgen import ProcgenGym3Env
#import pytorch_lightning as pl

from procgen import ProcgenEnv
#from algorithm.algorithm import Algorithm

#pl.seed_everything(123)

def run():
    env = ProcgenGym3Env(num=2, env_name="coinrun", render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")
    step = 0
    for i in range(100):
        
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1


if __name__ == '__main__':
    
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', default=0.02, type=float)
    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # parse params
    args = parser.parse_args()
    """
    run()
    
    