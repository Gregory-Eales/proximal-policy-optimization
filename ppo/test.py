import gym
import gym3
from tqdm import tqdm
from procgen import ProcgenGym3Env
"""
env = gym.make("CartPole-v1")
print("State: " ,env.reset().shape)
print("Actions", env.action_space)
print(" ")

env = gym.make("Acrobot-v1")
print("State: " ,env.reset().shape)
print("Actions", env.action_space)
print(" ")
"""
"""
env = gym.make("LunarLander-v2")
state = env.reset()

for i in tqdm(range(10000)):

	state, reward, done, info = env.step(env.action_space.sample())

	if done:
		env.reset()


env = gym.make("Amidar-ram-v4")

state = env.reset()

for i in tqdm(range(10000)):

	state, reward, done, info = env.step(env.action_space.sample())

	if done:
		env.reset()
"""
"""
env = gym.make("Assault-ram-v4")
print("State: " ,env.reset().shape)
print("Actions: ", env.action_space.sample())
print(" ")
"""

"""
env = gym.make("procgen:procgen-coinrun-v0",
	center_agent=False,
	render_mode="human")
"""

"""
import gym3
from procgen import ProcgenGym3Env

import procgen.env

env_names = procgen.env.ENV_NAMES



states = []
step = 0
for i in tqdm(range(100)):
	for name in env_names:

		env = ProcgenGym3Env(
			num=1,
			env_name=name,
			render_mode="rgb_array",
			center_agent=True,
			use_sequential_levels=False,
			distribution_mode="easy"
			)

		#env = gym3.ViewerWrapper(env, info_key="rgb")

		#env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
		rew, obs, first = env.observe()
		states.append(obs)
		#print(f"step {step} reward {rew} first {first}")
		#step += 1

print(len(states))
"""
env = ProcgenGym3Env(
			num=1,
			env_name="coinrun",
			render_mode="rgb_array",
			center_agent=False,
			num_levels=1,
			start_level=2,
			)

env = gym3.ViewerWrapper(env, info_key="rgb")

for i in tqdm(range(100)):
	

	

	env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
	rew, obs, first = env.observe()
	#states.append(obs)
	#print(f"step {step} reward {rew} first {first}")
	#step += 1