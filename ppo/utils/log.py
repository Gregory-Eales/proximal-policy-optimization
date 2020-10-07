import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')


print("Average Episode Length: {}".format(
    np.sum(episode_lengths)/len(episode_lengths)))

print("Largest Episode Length: {}".format(max(episode_lengths)))

print("Epoch: {}".format(epoch))
