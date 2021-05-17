#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2021
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Mini-Project 6

import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualizeNeighbors(imgs, topk_idxs, topk_distances, title):
	'''
	Visualize the query image as well as its nearest neighbors
	Input:
		imgs: a list or numpy array, with length k+1.
			imgs[0] is the query image with shape hxw
			imgs[k] is k-th nearest neighbor image
		topk_idxs: a list or numpy array, with length k+1.
			topk_idxs[k] is the index in training set of the k-th nearest image 
			topk_idxs[0] is the query image index in the test set
		topk_distances: a list or numpy array, with length k+1.
			topk_idxs[k] is the distance of the k-th nearest image to the query
			topk_idxs[0] is 0
	'''
	n = len(imgs)
	fig, axs = plt.subplots(1, n, figsize=(2 * n, 3))
	fig.suptitle(title)
	for k in range(n):
		if k == 0:
			ax_title = 'query: test_idx=%d' % topk_idxs[0]
		else:
			ax_title = '%d: idx=%d,d=%.2e' %(k, topk_idxs[k], topk_distances[k])
		axs[k].set_title(ax_title)
		axs[k].imshow(imgs[k], cmap='gray')
		axs[k].axis('off')
	fig.tight_layout()
	plt.show() 

	return 		


def knn(visualize):
	data = pickle.load(open('data_binarized.pkl','rb'))

	# distances = np.zeros((len(data['test']['y']), len(data['train']['y'])))

	# List of dictionary
	# element i-th is a dictionary with key [i-th, train img]
	# dict keys: [test img, train img]
	# dict value: Euclidean distance
	distances = [{}] * 200
	#--------------------------------------------------------------------------
	# Your implementation to calculate and sort distances
	#--------------------------------------------------------------------------
	x_train = data['train']['x']
	x_test = data['test']['x']
	for i in range(len(data['test']['y'])):		# loop through each image
		diction = {}
		for j in range(len(data['train']['y'])):
			# Apply Euclidean formula
			diction[i, j] = np.sqrt(np.sum(np.square(x_test[:, :, i] - x_train[:, :, j])))
		# Sort dictionary by values
		diction = dict(sorted(diction.items(), key=lambda item: item[1]))
		distances[i] = diction

	if visualize:
		k = 5
		imgs = np.random.randint(2, size=(k+1, 28, 28))
		topk_idxs = [0] * (k+1)
		topk_distances = [0] * (k+1)
		for test_i in [10, 20, 110, 120]:
			topk_idxs[0] = test_i
			# Assign query img as test_i
			imgs[0] = x_test[:, :, test_i]
			#------------------------------------------------------------------
			# Prepare imgs, topk_idxs and topk_distances
			#------------------------------------------------------------------
			i = 0
			for key in distances[test_i]:
				i += 1		# Only go through k nearest neighbor
				imgs[i] = x_test[:, :, key[1]]		# key[1] is index for training img of k-th neighbor
				topk_idxs[i] = key[1]
				topk_distances[i] = distances[test_i][key]
				if i >= k:
					break

			visualizeNeighbors(imgs, topk_idxs, topk_distances, 
				title='Test img %d: Top %d Neighbors' % (test_i, k))

	k_list = [1, 3, 5, 7, 9]
	accuracy_list = [0.0] * len(k_list)
	#--------------------------------------------------------------------------
	# Your implementation to calculate knn accuracy
	#--------------------------------------------------------------------------
	for k, acc in zip(k_list, accuracy_list):
		# Go through all test imgages
		for i in range(len(data['test']['y'])):
			five, nine, j = 0, 0, 0
			for key in distances[i]:
				j += 1		# Only check k nearest neighbor

				# Finding majority
				if data['train']['y'][key[1]] == 5:
					five += 1
				else:
					nine += 1
				if j >= k:
					break

			if five >= nine:		# 5 is majority
				if data['test']['y'][i] == 5:
					acc += 1
			else:					# 9 is majority
				if data['test']['y'][i] == 9:
					acc += 1
		# divide by total image (200) to get probability
		acc /= len(data['test']['y'])
		print('k=%d: accuracy=%.2f%%' % (k, acc * 100))

	return


if __name__ == "__main__":
	knn(visualize=True)