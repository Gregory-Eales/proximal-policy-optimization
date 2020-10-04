from matplotlib import pyplot as plt

plt.title("Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.plot(np.array(average_rewards), label="average reward")
plt.plot(highest_rewards, label="highest reward")
plt.legend(loc="upper left")
plt.draw()

if epoch%10 == 0:
	plt.savefig('reward_img/epoch{}.png'.format(epoch))

plt.pause(0.0001)
plt.clf()