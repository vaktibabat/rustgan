import numpy as np
import csv
import matplotlib.pyplot as plt

for num_epoch in range(0, 50):
	pixels = open("images/{}".format(num_epoch)).read()
	pixels = [255 * float(x.strip()) for x in pixels.split(",")[:len(pixels.split(","))-1]]
	pixels = np.array(pixels, dtype="uint8")
	pixels = pixels.reshape((28, 28))
	fig, ax = plt.subplots()
	#figManager = plt.get_current_fig_manager()
	#figManager.window.showMaximized()
	plt.title("Generated digit after epoch {}".format(num_epoch))
	ax.imshow(pixels, cmap="gray")
	plt.show(block=False)
	plt.pause(1)
	plt.close(fig)
	#input("Continue to the next image? ")
