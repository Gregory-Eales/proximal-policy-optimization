import logging

def init_log(path):
	logging.basicConfig(filename='example.log', level=logging.DEBUG)

def log():
	logging.basicConfig(filename='example.log', level=logging.DEBUG)
	logging.debug('This message should go to the log file')
	logging.info('So should this')
	logging.warning('And this, too')


	print("Average Episode Length: {}".format(
	    np.sum(episode_lengths)/len(episode_lengths)))

	print("Episode finished after {} timesteps".format(step+1))
	print("Largest Episode Length: {}".format(max(episode_lengths)))

	print("Epoch: {}".format(epoch))
