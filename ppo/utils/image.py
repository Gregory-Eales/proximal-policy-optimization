import imageio


def image():
	images = []
	for filename in filenames:
	    images.append(imageio.imread(filename))
	imageio.mimsave('/path/to/movie.gif', images)


	with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
	    for filename in filenames:
	        image = imageio.imread(filename)
	        writer.append_data(image)

	img = env.render(mode="rgb_array")
	scipy.misc.imsave('img/gif/img{}.jpg'.format(frame), img)
