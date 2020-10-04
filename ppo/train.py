from argparse import ArgumentParser

import modules
import utils

def main(params):

    agent = PPOAgent(param)


if __name__ == '__main__':
    
    
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', default="default", type=str)
    parser.add_argument('--environment name', default="couinrun")

    parser.add_argument('--actor_lr', default=2e-1, type=float)
    parser.add_argument('--critic_lr', default=2e-1, type=float)
    
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--n_steps', default=200, type=int)

    parser.add_argument('--verbos', default=True, type=bool)
    parser.add_argument('--graph', default=True, type=bool)

    parser.add_argument('random_seeds', default=list(range(10)), type=list)

    parser.add_argument('--learning_rate', default=0.02, type=float)
    
    

    params = parser.parse_args()

    main(params)
    
    