# -*- encoding=UTF-8-*-

import sys
import numpy as np
from skimage.util import img_as_float
#import matplotlib.patches as patches
#from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import io, exposure

import models
import image_utils

image_width = 256
image_height = 256
channels = 4
extension = '.tif'
classes = {
    'pivot': 0,
    'infra': 1,
}
num_classes = len(classes)
weights_path = "weights"

def train(model, epochs=200, batch_size=100):
    features, target = image_utils.load_data("data/train", classes, image_width, image_height, extension, dargumentation_enabled=True)
    #train_data, eval_data, train_labels, eval_labels = train_test_split(features, target, test_size=0.2)

    #print('Split train: ', len(train_data))
    #print('Split test: ', len(eval_data))

    classifier = tf.estimator.Estimator(model_fn=model, model_dir=weights_path)
    logging_hook = tf.train.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=np.asarray(target, dtype=np.int32),
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True
    )

    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": eval_data},
    #    y=np.asarray(eval_labels, dtype=np.int32),
    #    num_epochs=1,
    #   shuffle=False)

    #for epoch in range(1, epochs + 1):
    #print("EPOCH:", epoch)
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
    #eval_results = classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)

def evaluate(model):
    validation_features, validation_target = image_utils.load_data("data/validation", classes, image_width, image_height, extension)

    classifier = tf.estimator.Estimator(model_fn=model, model_dir=weights_path)

    validation_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": validation_features},
        y=np.asarray(validation_target, dtype=np.int32),
        num_epochs=1,
        shuffle=False)

    validation_results = classifier.evaluate(input_fn=validation_input_fn)
    print(validation_results)


def predict(model, path):
    stepSize = 100
    batch_size = 5
    windows_list = (
        #[int(image_width * 0.25), int(image_height * 0.25)],
        #[int(image_width * 0.5), int(image_height * 0.5)],
        [image_width, image_height],
        #[int(image_width * 1.5), int(image_height * 1.5)],
        #[int(image_width * 1.75), int(image_height * 1.75)],
    )

    classifier = tf.estimator.Estimator(model_fn=model, model_dir=weights_path)

    image = image_utils.load_file(path)
    image = image[:, :, :channels]

    plt.ion()
    fig, ax = plt.subplots(1)
    image_rgb = img_as_float(io.imread(path))
    image_rgb = image_rgb[:, :, :3]
    image_rgb = exposure.rescale_intensity(image_rgb)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    ax.imshow(image_rgb)

    batch = []
    count = 1

    for width, height in windows_list:
        print(width, height)
        for (x, y, window) in image_utils.sliding_window(image,
                                                         stepSize=stepSize,
                                                         windowSize=(width, height),
                                                         windowResize=(image_width, image_height)):

            if window.shape[0] != image_height or window.shape[1] != image_width:
                continue

            batch.append({
                "window": window,
                "x": x,
                "y": y
            })

            if len(batch) >= batch_size:
                fig.savefig('results/{0}.png'.format(count))
                count += 1

                windows = []
                positions = []
                for b in batch:
                    windows.append(b.get("window"))
                    positions.append((b.get("x"), b.get("y")))
                windows = np.array(windows)

                predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": np.asarray(windows, dtype=np.float32)},
                    shuffle=False,
                    num_epochs=1,
                )

                results = classifier.predict(input_fn=predict_input_fn)
                pred = []
                pred_prob = []
                for p in results:
                    print(p)
                    pred.append(p["classes"])
                    pred_prob.append(p["probabilities"])

                for window, position, label, prob in zip(windows, positions, pred, pred_prob):
                    prediction = prob[np.argmax(prob)]
                    print("Label: {0} --> Prediction: {1}".format(label, prediction))

                    if label == classes.get("pivot") and prediction > 0.50:
                        rect = patches.Rectangle((position[0], position[1]), width, height, linewidth=1,
                                                 edgecolor='r', color='y', facecolor='none')
                    else:
                        rect = patches.Rectangle((position[0], position[1]), width, height, linewidth=1,
                                                 edgecolor='r', facecolor='none')

                    ax.add_patch(rect)

                batch = []

            # window = window[-1][:,:,:3]
            # print window.shape
            # plt.axis("off")
            # plt.imshow(window.reshape((window.shape[0], window.shape[1])), cmap=plt.cm.nipy_spectral, interpolation='nearest')
            # plt.show()
            fig.canvas.draw()


mode = sys.argv[1]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model = models.get_model(image_width, image_height, channels, num_classes, learning_rate=0.00001)

    if mode == 'train':
        train(model)
    elif mode == 'evaluate':
        evaluate(model)
    elif mode == 'predict':
        file_path = sys.argv[2]
        predict(model, file_path)
