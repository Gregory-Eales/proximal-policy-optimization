from argparse import ArgumentParser
import os

import modules
import utils


def train(
    agent,
    env,
    n_epoch,
    n_steps,
    ):

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

            

            
            # for t steps:
            for t in range(n_steps):

                # increment step
                step += 1

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
            
        
            
          

def run_experiment(
    experiment_name,
    environement_name,
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

    agent = PPO()






if __name__ == '__main__':
    
    
    parser = ArgumentParser(add_help=False)

    # experiment and  environment
    parser.add_argument('--experiment_name', default="default", type=str)
    parser.add_argument('--environment name', default="couinrun")

    # saving options
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--graph', default=True, type=bool)

    # training params
    parser.add_argument('--random_seeds', default=list(range(10)), type=list)
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--n_steps', default=200, type=int)
    parser.add_argument('--batch_sz', default=16, type=int)
    parser.add_argument('--gamma', default=16, type=int)
    parser.add_argument('--batch_sz', default=16, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--training_epochs', default=10, type=int)

    # model params
    parser.add_argument('--actor_lr', default=2e-1, type=float)
    parser.add_argument('--critic_lr', default=2e-1, type=float)
    parser.add_argument('--epsilon', default=0.3, type=float)
    

    params = parser.parse_args()

    #run_experiment(params)
    
    