from matplotlib import pyplot as plt

import numpy as np


def generate_graphs(agent, path=""):

	reward_per_epoch(agent.get_rewards(), path=path)

def reward_per_epoch(reward, path=""):

	plt.title("Reward per Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.plot(reward, label="reward")
	plt.legend(loc="upper left")
	plt.savefig('{}/graphs/reward_per_epoch.png'.format(path))


	"""
	CONFIDENCE BANDS !
	
	N = 21
	x = np.linspace(0, 10, 11)
	y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

	# fit a linear curve an estimate its y-values and their error.
	a, b = np.polyfit(x, y, deg=1)
	y_est = a * x + b
	y_err = x.std() * np.sqrt(1/len(x) +
	                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

	fig, ax = plt.subplots()
	ax.plot(x, y_est, '-')
	ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
	ax.plot(x, y, 'o', color='tab:brown')
	"""

def main():

	reward_per_epoch(np.random.random([2, 10]))


if __name__ == "__main__":
	main()