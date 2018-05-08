import numpy as np
import math
import tensorflow as tf
import cv2
import os
import shutil
import glob
import csv



class Face:
	'''A face from the Chicago Face Database, together with an attractiveness score
	Attractiveness is normalized to [0, 1]'''
	def __init__(self, name, attractiveness):
		self.name = name
		self.attractiveness = attractiveness # scaled to be in [0, 1]


	def get_image(self, dims = None):
		image_regex = os.path.abspath('./CFD Version 2.0.3/CFD 2.0.3 Images/' + self.name + '/*.jpg')
		image_filepaths = [f for f in glob.iglob(image_regex)]
		image_filapath = np.random.choice(image_filepaths) # some faces have several photos of them, choose one at random
		image = cv2.cvtColor(cv2.imread(image_filapath), cv2.COLOR_BGR2RGB) / 255
		return cut_resize_image(image=image, dims=dims)



faces = []
with open('./CFD_data.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ',')
	for row in reader:
		attractiveness = (float(row[15].replace(',', '.')) - 1) / 5
		face = Face(name = row[0], attractiveness = attractiveness)
		faces.append(face)

np.random.seed(0)
np.random.shuffle(faces)
test_size = 64
train_faces = faces[:-test_size]
test_faces = faces[-test_size:]
del faces


def generator(batch_function, batch_size = 1, dims = None, test = False):
	'''Generator that keeps indefinitely calling batch_function'''
	batch_number = 0
	while True:
		yield batch_function(batch_size, dims = dims, test = test, batch_number = batch_number)
		batch_number += 1
		if batch_number % set_steps(batch_size = batch_size, test = test) == 0:
			np.random.shuffle(get_set(test = test))


def attractiveness_batch(batch_size = None, dims = None, test = False, batch_number = None):
	'''Batch of face images with attractiveness ratings as targets'''
	faces = face_batch(batch_size = batch_size, test = test, batch_number = batch_number)
	return np.array([face.get_image(dims) for face in faces]), np.array([face.attractiveness for face in faces])


def auto_batch(batch_size = None, dims = None, test = False, batch_number = None):
	'''Batch of face images with themselves as targets'''
	faces = face_batch(batch_size = batch_size, test = test, batch_number = batch_number)
	return np.array([face.get_image(dims) for face in faces]), np.array([face.attractiveness for face in faces])


def face_batch(batch_size = None, test = False, batch_number = None):
	'''Batch of Face instances chosen from appropriate set'''
	set = get_set(test)
	if not batch_size:
		batch_size = len(set)
	if batch_number:
		start_index = batch_size * (batch_number % set_steps(batch_size = batch_size, test = test))
		end_index = min(start_index + batch_size, len(set))
		return set[start_index : end_index]
	return np.random.choice(set, batch_size)


def set_steps(batch_size, test):
	return math.ceil(len(test_faces) if test else len(train_faces) / batch_size)


def get_set(test):
	return test_faces if test else train_faces


def image_summary(name, tensor, image_dims, channels = 3, max_images = 32, tf_normalize = True):
	if tf_normalize:
		tf.summary.image(name, tf.reshape(tensor, (-1, image_dims[0], image_dims[1], channels)), max_outputs = max_images)
	else:
		tf.summary.image(name, tf.cast(tf.reshape(tensor, (-1, image_dims[0], image_dims[1], channels)) * 255.5, tf.uint8), max_outputs = max_images)


def preview(names, images, channels = 3, milliseconds = 0, destroy = True):
	for i in range(len(names)):
		colored = cv2.cvtColor(images[i].astype(np.float32), cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_GRAY2BGR)
		cv2.imshow(names[i], cv2.resize(colored, (256, 256), interpolation = cv2.INTER_NEAREST))

	key = cv2.waitKey(milliseconds)

	if destroy:
		for i in range(len(names)):
			cv2.destroyWindow(names[i])
	return key



def cut_resize_image(image, dims = None):
	'''Cut & resize so as to lose as little as possible while preserving aspect ratio'''
	if not dims:
		return image
	dims_ratio = dims[0] / dims[1]
	image_ratio = len(image) / len(image[0])
	if dims_ratio > image_ratio:
		width = len(image) * dims_ratio
		mid = len(image[0]) / 2
		return cv2.resize(image[:, int(mid - width / 2) : int(mid + width / 2)], dims,
		interpolation = cv2.INTER_AREA if width < len(image[0]) else cv2.INTER_NEAREST)
	elif dims_ratio < image_ratio:
		height = len(image[0]) / dims_ratio
		mid = len(image) / 2
		return cv2.resize(image[int(mid - height / 2) : int(mid + height / 2)], dims,
		interpolation = cv2.INTER_AREA if height < len(image) else cv2.INTER_NEAREST)
	return cv2.resize(image, dims,
	interpolation = cv2.INTER_AREA if dims[0] < len(image) else cv2.INTER_NEAREST)



def try_remove_dir(relative_dir):
	shutil.rmtree(os.path.join(os.path.abspath('./'), relative_dir), ignore_errors = True)



def leaky_relu(tensor, leak = 0.1):
	return tf.maximum(tensor, tensor * leak)



def shape_volume(shape):
	result = 1
	for dim in shape:
		result *= dim
	return result
