import torch
from procgen import ProcgenEnv


class Trainer(object):

    def __init__(self):
        pass

    def __call__(self, params, agent, env):

        torch.manual_seed(1)
        np.random.seed(1)


        env = ProcgenEnv(env_name="coinrun", render_mode="rgb_array")
        step = 0
        for i in range(100):
            
            env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
            rew, obs, first = env.observe()
            print(f"step {step} reward {rew} first {first}")
            step += 1
