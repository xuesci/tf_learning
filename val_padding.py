# coding:utf-8
import tensorflow as tf
import numpy as np

# tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(1, 50, 50, 1))
W1 = tf.get_variable('W1', shape=[3, 3, 1, 1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))
W2 = tf.get_variable('W2', shape=[5, 5, 1, 1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))
W3 = tf.get_variable('W3', shape=[7, 7, 1, 1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))
W4 = tf.get_variable('W4', shape=[9, 9, 1, 1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))
W5 = tf.get_variable('W5', shape=[11, 11, 1, 1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))

conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
conv2_same = tf.nn.conv2d(x, W2, strides=[1, 2, 2, 1], padding='SAME')
conv2_same_1 = tf.nn.conv2d(conv2_same, W2, strides=[1, 2, 2, 1], padding='SAME')
conv2_same_2 = tf.nn.conv2d(conv2_same_1, W2, strides=[1, 2, 2, 1], padding='SAME')
conv2_same_3 = tf.nn.conv2d(conv2_same_2, W2, strides=[1, 2, 2, 1], padding='SAME')

conv2_valid = tf.nn.conv2d(x, W2, strides=[1, 2, 2, 1], padding='VALID')
conv2_valid_1 = tf.nn.conv2d(conv2_valid, W2, strides=[1, 2, 2, 1], padding='VALID')
conv2_valid_2 = tf.nn.conv2d(conv2_valid_1, W2, strides=[1, 2, 2, 1], padding='VALID')

with tf.Session() as sess:
	feed_data = np.full((1, 50, 50, 1), 1)
	sess.run(tf.global_variables_initializer())
	[conv2_same, conv2_same_1, conv2_same_2, conv2_same_3] = sess.run([conv2_same, conv2_same_1, conv2_same_2, conv2_same_3], feed_dict={x: feed_data})
	[conv2_valid, conv2_valid_1, conv2_valid_2] = sess.run([conv2_valid, conv2_valid_1, conv2_valid_2], feed_dict={x: feed_data})

	conv2_same = conv2_same[0]
	conv2_same_map = []
	for i in range(len(conv2_same)):
		for j in range(len(conv2_same)):
			conv2_same_map.append(conv2_same[i][j][0])
	width = height = int(np.sqrt(len(conv2_same_map)))
	conv2_same = np.reshape(conv2_same_map, (width, height))
	print("第一次卷积后特征图的尺寸：", len(conv2_same), "×", len(conv2_same))
	# print(conv2_same)
	print("***************************************************************")

	conv2_same_1 = conv2_same_1[0]
	conv2_same_1_map = []
	for i in range(len(conv2_same_1)):
		for j in range(len(conv2_same_1)):
			conv2_same_1_map.append(conv2_same_1[i][j][0])
	width = height = int(np.sqrt(len(conv2_same_1_map)))
	conv2_same_1 = np.reshape(conv2_same_1_map, (width, height))
	print("第二次卷积后特征图的尺寸：", len(conv2_same_1), "×", len(conv2_same_1))
	# print(conv2_same_1)
	print("***************************************************************")

	conv2_same_2 = conv2_same_2[0]
	conv2_same_2_map = []
	for i in range(len(conv2_same_2)):
		for j in range(len(conv2_same_2)):
			conv2_same_2_map.append(conv2_same_2[i][j][0])
	width = height = int(np.sqrt(len(conv2_same_2_map)))
	conv2_same_2 = np.reshape(conv2_same_2_map, (width, height))
	print("第三次卷积后特征图的尺寸：", len(conv2_same_2), "×", len(conv2_same_2))
	# print(conv2_same_2)
	print("***************************************************************")

	conv2_same_3 = conv2_same_3[0]
	conv2_same_3_map = []
	for i in range(len(conv2_same_3)):
		for j in range(len(conv2_same_3)):
			conv2_same_3_map.append(conv2_same_3[i][j][0])
	width = height = int(np.sqrt(len(conv2_same_3_map)))
	conv2_same_3 = np.reshape(conv2_same_3_map, (width, height))
	print("第四次卷积后特征图的尺寸：", len(conv2_same_3), "×", len(conv2_same_3))
	# print(conv2_same_3)
	print("***************************************************************\n")

	conv2_valid = conv2_valid[0]
	conv2_valid_map = []
	for i in range(len(conv2_valid)):
		for j in range(len(conv2_valid)):
			conv2_valid_map.append(conv2_valid[i][j][0])
	width = height = int(np.sqrt(len(conv2_valid_map)))
	conv2_valid = np.reshape(conv2_valid_map, (width, height))
	print("第一次卷积后特征图的尺寸：", len(conv2_valid), "×", len(conv2_valid))
	# print(conv2_valid)
	print("***************************************************************")

	conv2_valid_1 = conv2_valid_1[0]
	conv2_valid_1_map = []
	for i in range(len(conv2_valid_1)):
		for j in range(len(conv2_valid_1)):
			conv2_valid_1_map.append(conv2_valid_1[i][j][0])
	width = height = int(np.sqrt(len(conv2_valid_1_map)))
	conv2_valid_1 = np.reshape(conv2_valid_1_map, (width, height))
	print("第二次卷积后特征图的尺寸：", len(conv2_valid_1), "×", len(conv2_valid_1))
	# print(conv2_valid_1)
	print("***************************************************************")

	conv2_valid_2 = conv2_valid_2[0]
	conv2_valid_2_map = []
	for i in range(len(conv2_valid_2)):
		for j in range(len(conv2_valid_2)):
			conv2_valid_2_map.append(conv2_valid_2[i][j][0])
	width = height = int(np.sqrt(len(conv2_valid_2_map)))
	conv2_valid_2 = np.reshape(conv2_valid_2_map, (width, height))
	print("第三次卷积后特征图的尺寸：", len(conv2_valid_2), "×", len(conv2_valid_2))
	# print(conv2_valid_2)
	print("***************************************************************")