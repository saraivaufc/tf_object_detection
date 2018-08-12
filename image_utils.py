# -*- encoding=UTF-8-*-

import os
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
from sklearn import decomposition
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.transform import resize, rotate
from skimage import exposure
from tqdm import tqdm

SCALE_LEVEL = 0.60


def load_file(path, width=None, height=None):
	dataSource = gdal.Open(path)

	bands = []
	for index in range(1, dataSource.RasterCount + 1):
		band = dataSource.GetRasterBand(index).ReadAsArray()
		band_min = np.min(band)
		band_max = np.max(band)
		band_normalized = (2 * ((band - band_min) / float(band_max - band_min))) - 1
		bands.append(band_normalized)

	image = np.dstack(bands)

	if width and height:
		image = resize(image, (width, height))

	# plt.imshow(image[: , : , 0:3])
	# plt.show()

	return image


def dargumentation(image):
	images = []
	for rot in [0, 90, 180, 270]:
		images.append(rotate(image, rot))
	return images


def load_data(directory, labels, width, height, extension):
	npz_path = directory + '.npz'

	if os.path.isfile(npz_path):
		print("Reading from cache " + npz_path + "...")
		data = np.load(npz_path)
		input_data = list(zip(*data['input']))

		x = np.array(input_data[0])
		y = np.array(input_data[1])

		return x, y
	else:
		x = []
		y = []
		for label, label_value in labels.items():
			label_directory = "{root_path}/{label}".format(root_path=directory, label=label)
			files = [f for f in os.listdir(label_directory) if os.path.isfile(os.path.join(label_directory, f)) and f.endswith(extension)]
			for f in tqdm(files, miniters=10):
				path = "{directory}/{file}".format(directory=label_directory, file=f)
				image = load_file(path, width, height)
				
				for i in dargumentation(image):
					x.append(i)
					y.append(label_value)

		x = np.array(x, np.float32)
		y = np.array(y, np.uint8)
		data = np.array(list(zip(x, y)))
		np.savez_compressed(npz_path, input=data)
		return x, y

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			window = image[y:y + windowSize[1], x:x + windowSize[0]]
			# window = resize(window, (windowSize[0], windowSize[1]))
			yield (x, y, window)