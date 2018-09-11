# -*- encoding=UTF-8-*-

import tensorflow as tf

learning_rate = 0.0005


def get_model(width, height, channels, num_classes):
	def model_fn(features, labels, mode):
		"""Model function for CNN."""
		# Input Layer
		input_layer = tf.reshape(features["x"], [-1, width, height, channels])

		conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv1 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv1_norm = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
		pool1 = tf.layers.max_pooling2d(inputs=conv1_norm, pool_size=[2, 2], strides=2)

		conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv2_norm = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
		pool2 = tf.layers.max_pooling2d(inputs=conv2_norm, pool_size=[2, 2], strides=2)

		conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv3_norm = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
		pool3 = tf.layers.max_pooling2d(inputs=conv3_norm, pool_size=[2, 2], strides=2)

		conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv4 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv4_norm = tf.layers.batch_normalization(conv4, training=(mode == tf.estimator.ModeKeys.TRAIN))
		pool4 = tf.layers.max_pooling2d(inputs=conv4_norm, pool_size=[2, 2], strides=2)

		conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
		conv5_norm = tf.layers.batch_normalization(conv5, training=(mode == tf.estimator.ModeKeys.TRAIN))
		pool5 = tf.layers.max_pooling2d(inputs=conv5_norm, pool_size=[2, 2], strides=2)

		output = pool5

		# Dense Layer
		flat = tf.reshape(output, [-1, output.shape[1] * output.shape[2] * output.shape[3]])

		dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
		BN = tf.layers.batch_normalization(dense1, training=(mode == tf.estimator.ModeKeys.TRAIN))
		dropout = tf.layers.dropout(inputs=BN, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
	
		# Logits Layer
		logits = tf.layers.dense(inputs=dropout, units=num_classes)

		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.contrib.opt.NadamOptimizer(learning_rate, name='optimizer')
			train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

			with tf.name_scope("img_metrics"):
				input_data_viz = ((features["x"][:, :, :, 0:3]) + 64)
				input_data_viz = tf.image.convert_image_dtype(input_data_viz, tf.uint8)
				tf.summary.image('img', input_data_viz, max_outputs=1)

			eval_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir="eval", summary_op=tf.summary.merge_all())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, evaluation_hooks=[eval_summary_hook])

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	return model_fn
