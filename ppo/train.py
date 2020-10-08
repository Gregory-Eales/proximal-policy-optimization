from argparse import ArgumentParser
from tqdm import tqdm
import os
import random
import gym
import gym3
import time

from modules import *
from utils import *


def train(agent, env, n_epoch, n_steps):

        for epoch in tqdm(range(n_epoch)):

            prev_state = env.reset()
            done = False

            while not done:

                env.render("human")

                action = agent.act(prev_state)

                state, reward, done, info = env.step(action)

                agent.store(action, state, reward, prev_state)

                prev_state = state

                if done:
                    prev_state = env.reset()


            agent.update()

        

def run_experiment(
    experiment_name,
    environment_name,
    log,
    graph,
    random_seeds,
    n_episodes,
    n_steps,
    actor_lr,
    critic_lr,
    epsilon
    ):
    
    """

    experiment_name: name of the experiment

    environment_name: env to be used for the experiment

    logging: whether or not to log True/False

    graph: whether or not to graph True/False

    random_seeds: random seeds to use in the experiment

    n_episodes: number of complete episodes
    
    n_steps: number of steps per episode

    actor_lr: actor learning rate

    critic_lr: critic learning rate

    epsilon: the clip range for the ppo objective

    """

    # setup experiment path

    exp_path = create_exp_dir(experiment_name)

    #agent = PPO()
    #env = ProcgenGym3Env(num=1, env_name="coinrun", render_mode="rgb_array")
    #env = gym3.ViewerWrapper(env, info_key="rgb")
    agent = Agent()
    
    env = gym.make("procgen:procgen-coinrun-v0")
    
    t = time.time()
    train(agent, env, n_episodes, n_steps)
    print(time.time()-t)

    print(len(agent.reward))

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
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--n_steps', default=1000, type=int)
    parser.add_argument('--batch_sz', default=16, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--training_epochs', default=10, type=int)

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
        actor_lr=params.actor_lr,
        critic_lr=params.critic_lr,
        epsilon=params.epsilon
        )
    
    