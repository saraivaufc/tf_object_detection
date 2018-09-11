# -*- encoding=UTF-8-*-

import os
import numpy as np
from osgeo import gdal
# from matplotlib import pyplot as plt
from sklearn import decomposition
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.transform import resize, rotate
from skimage import exposure
from tqdm import tqdm
import h5py


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

    return image


def dargumentation(image):
    images = []
    for rot in [0, 90, 180, 270]:
        images.append(rotate(image, rot))
    return images


def load_data(directory, classes, width, height, extension, dargumentation_enabled=False):
    data_path = directory + '.h5'

    if os.path.isfile(data_path):
        print("Reading from cache " + data_path + "...")

        with h5py.File(data_path, 'r') as hf:
            x = hf['x'][:]
            y = hf['y'][:]
        print("Completed!!!")
        return x, y
    else:
        x = []
        y = []
        for label, label_value in classes.items():
            label_directory = "{root_path}/{label}".format(root_path=directory, label=label)
            files = [f for f in os.listdir(label_directory) if os.path.isfile(os.path.join(label_directory, f)) and f.endswith(extension)]
            for f in tqdm(files, miniters=10):
                path = "{directory}/{file}".format(directory=label_directory, file=f)
                image = load_file(path, width, height)
                if dargumentation_enabled:
                    for i in dargumentation(image):
                        x.append(i)
                        y.append(label_value)
                else:
                    x.append(image)
                    y.append(label_value)


        x = np.array(x, np.float32)
        y = np.array(y, np.uint8)

        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset("x", data=x)
            hf.create_dataset("y", data=y)
        print("Completed!!!")
        return x, y


def sliding_window(image, stepSize, windowSize, windowResize=None):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            window = image[y:y + windowSize[1], x:x + windowSize[0]]
            if(windowResize):
                window = resize(window, (windowResize[0], windowResize[1]))
            yield (x, y, window)
