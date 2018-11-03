import imageio

images = []
idx = 1
idx_max = 1000
while idx < idx_max:
	filename = "images/plot_epoch_{0:03}_generated.png".format(idx)
	print(filename)
	images.append(imageio.imread(filename))
	idx += 1

imageio.mimsave('progression.gif', images)
