import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
import os
import math
import cv2
import pickle

from .norb_reader import *

def place_random(trainX):
	#randomly place 28x28 mnist image on 40x40 background
	trainX_new = []
	for img in trainX:
		x_len = maxX - minX
		y_len = maxY - minY

		img_new = np.zeros((40,40,1), dtype=np.float32)
		x = np.random.randint(12 , size=1)[0]
		y = np.random.randint(12 , size=1)[0]

		img_new[y:y+28, x:x+28, :] = img
		trainX_new.append(img_new)
	
	return np.array(trainX_new)

def one_hot(label, output_dim):
	one_hot = np.zeros((len(label), output_dim))
	
	for idx in range(0,len(label)):
		one_hot[idx, label[idx]] = 1
	
	return one_hot

def load_data_from_mat(path):
	data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
	for key in data:
		if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
			data[key] = _todict(data[key])
	return data

def _todict(matobj):
    #A recursive function which constructs from matobjects nested dictionaries
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#============== Different Readers ==============

def affnist_reader(args, path):
	train_path = glob(os.path.join(path, "train/*.mat"))
	test_path = glob(os.path.join(path, "test/*.mat"))

	train_data = load_data_from_mat(train_path[0])

	trainX = train_data['affNISTdata']['image'].transpose()
	trainY = train_data['affNISTdata']['label_int']

	trainX = trainX.reshape((50000, 40, 40, 1)).astype(np.float32)
	trainY = trainY.reshape((50000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)

	test_data = load_data_from_mat(test_path[0])
	testX = test_data['affNISTdata']['image'].transpose()
	testY = test_data['affNISTdata']['label_int']

	testX = testX.reshape((10000, 40, 40, 1)).astype(np.float32)
	testY = testY.reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)	

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-60, maxval=60, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count

def cifar_reader(args, path):
	def unpickle(file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo)
		return dict

	train_path = glob(os.path.join(path, "data_batch_*"))	
	test_path = glob(os.path.join(path, "test_batch"))

	trainX = []
	trainY = []
	for p in train_path:
		extracted = unpickle(p)
		image = None
		for t in extracted['data']:
			r = t[:1024].reshape(32,32,1)
			g = t[1024:2048].reshape(32,32,1)
			b = t[2048:].reshape(32,32,1)
			image = np.concatenate([r,g,b], axis=2)
			trainX.append(image)
		trainY.append(extracted['labels'])

	trainX = np.array(trainX)	
	trainY = np.concatenate(np.array(trainY), axis=0)	
	trainY = one_hot(trainY, args.output_dim)

	testX = []
	testY = []
	extracted = unpickle(test_path[0])
	for t in extracted['data']:
		r = t[:1024].reshape(32,32,1)
		g = t[1024:2048].reshape(32,32,1)
		b = t[2048:].reshape(32,32,1)
		image = np.concatenate([r,g,b], axis=2)
		testX.append(image)
	testY.append(extracted['labels'])
	testX = np.array(testX)	
	testY = np.concatenate(np.array(testY), axis=0)		
	testY = one_hot(testY, args.output_dim)


	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)		

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count


def small_norb_reader(args, path):
	def extract_patch(dataset):
		extracted = []
		for img in train_dat:
			img_ = img[0].reshape(96,96,1)
			extracted.append(img_)
			img_ = img[1].reshape(96,96,1)
			extracted.append(img_)					
		return np.array(extracted)
	
	#Training Data
	file_handle = open(getPath('train','dat'))
	train_dat = parseNORBFile(file_handle)
	file_handle = open(getPath('train','cat'))
	train_cat = parseNORBFile(file_handle)

	#Test Data
	file_handle = open(getPath('test','dat'))
	test_dat = parseNORBFile(file_handle)
	file_handle = open(getPath('test','cat'))
	test_cat = parseNORBFile(file_handle)

	trainX = extract_patch(train_dat)
	trainY = np.repeat(train_cat, 2)
	trainY = one_hot(trainY, args.output_dim)

	testX = extract_patch(test_dat)
	testY = np.repeat(test_cat, 2)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)

		data_count = len(testX)		

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
		
	if args.is_train:	
		images = tf.image.resize_images(input_queue[0] ,[48, 48])
		images = tf.random_crop(images, [32, 32, 1])
	else:
		images = tf.image.resize_images(input_queue[0] ,[48, 48])
		images = tf.image.resize_image_with_crop_or_pad(images, 32, 32)

	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count

def yelp_reader(args, path):
	import pandas as pd
	from scipy import sparse as sp_sparse
	reviews = pd.read_csv('~/Downloads/yelp/yelp-dataset/yelp_review.csv', nrows=200) 
	X_train = reviews['text']
	y_train = reviews['stars']
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9, random_state=42)
	words = []
	for sentence in X_train:
	  words.extend(list((sentence.split())))
	words_unique = list(set(words))
	from collections import Counter
	words_counts = Counter(words)
	most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:5000]

	DICT_SIZE = 5000
	WORDS_TO_INDEX = {k:v for k, v in most_common_words}
	INDEX_TO_WORDS = {v:k for k, v in most_common_words}
	ALL_WORDS = WORDS_TO_INDEX.keys()

	def my_bag_of_words(text, words_to_index, dict_size):
	    """
	        text: a string
	        dict_size: size of the dictionary
	        
	        return a vector which is a bag-of-words representation of 'text'
	    """
	    result_vector = np.zeros(dict_size)
	    for word in text.split():
	      if word in words_to_index:
	        result_vector[words_to_index[word]] = result_vector[words_to_index[word]] + 1
	    return result_vector

	X_train_mybag = [my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE) for text in X_train]
	max_document_length = max([len(x.split(" ")) for x in X_train])
	import tensorflow as tf
	from tensorflow.contrib import lookup
	from tensorflow.python.platform import gfile

	MAX_DOCUMENT_LENGTH = 5  
	PADWORD = 'ZYXW'

	# create vocabulary
	vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
	vocab_processor.fit(X_train)
	X_train = np.array(list(vocab_processor.fit_transform(X_train)))
	with gfile.Open('vocab.tsv', 'wb') as f:
	    f.write("{}\n".format(PADWORD))
	    for word, index in vocab_processor.vocabulary_._mapping.items():
	      f.write("{}\n".format(word))
	N_WORDS = len(vocab_processor.vocabulary_)
	from sklearn import preprocessing
	lb = preprocessing.LabelBinarizer()
	lb.fit([1,2,3,4,5])
	y_train = lb.transform(y_train)
	y_test = lb.transform(y_test)
	trainX, trainY = X_train_mybag, y_train
	# Parameters
	if args.is_train:
		sequence_length=X_train.shape[1]
		num_classes=y_train.shape[1]
		vocab_size=len(vocab_processor.vocabulary_)
		embedding_size=128
		input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		W = tf.Variable(
		          tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
		          name="W")
		embedded_chars = tf.nn.embedding_lookup(W, input_x)
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
		X, Y, data_count = embedded_chars_expanded, tf.convert_to_tensor(trainY, dtype=tf.float32), len(X_train)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)		
	return X, Y, data_count
 
def fashion_mnist_reader(args, path):
	#Training Data
	f = open(os.path.join(path, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)


	#Test Data
	f = open(os.path.join(path, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)		

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )
	return X, Y, data_count


def mnist_reader(args, path):
	#Training Data
	f = open(os.path.join(path, 'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	if args.random_pos:
		trainX = place_random(trainX)

	f = open(os.path.join(path, 'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)


	#Test Data
	f = open(os.path.join(path, 't10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 't10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)

	input_queue = tf.train.slice_input_producer([X, Y], shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-60, maxval=60, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)


	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count




#load different datasets
def load_data(args):

	path = os.path.join(args.root_path, args.data)

	if args.data == "mnist":
		images, labels, data_count = mnist_reader(args, path)
	elif args.data == "fashion_mnist":
		images, labels, data_count = fashion_mnist_reader(args, path)
	elif args.data == "affnist":
		images, labels, data_count = affnist_reader(args, path)					
	elif args.data == "small_norb":
		images, labels, data_count = small_norb_reader(args, path)
	elif args.data == "cifar10":
		images, labels, data_count = cifar_reader(args, path)		
	elif args.data == "yelp":
		images, labels, data_count = yelp_reader(args, path)
	else:
		print("Invalid dataset name!!")

	return images, labels, data_count 