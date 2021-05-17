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


def scoreFeatures(x, y):
	scores = np.zeros(x.shape[:2])
	five, nine = np.zeros(x.shape[:2]), np.zeros(x.shape[:2])
	#--------------------------------------------------------------------------
	# Calculate scores (Implement this)
	#--------------------------------------------------------------------------

	# Calculate each attribute
	for i in range(len(x[:, :, 1])):		# loop through each pixel feature
		for j in range(len(x[i, :, 1])):
			for img in range(len(x[i, j])):		# loop through each image
				if x[i, j, img] == 0:		# 1 is on and 0 is off
					if y[img] == 5:			# 5 is 'hated'
						five[i, j] += 1
				else:
					if y[img] == 9:			# 9 is 'liked'
						nine[i, j] += 1

	scores = five + nine
	plt.imshow(scores, cmap='gray')
	plt.axis('off')
	plt.show()
	return scores


def highest_score(scores):
	r_i, r_j = 0, 0
	highest = 0
	for i in range(scores.shape[0]):
		for j in range(scores.shape[1]):
			check = highest
			highest = max(highest, scores[i, j])
			if highest != check:
				r_i, r_j = i, j
	print(r_i, r_j, highest)
	return r_i, r_j


def decisiontree_test(x, y, i, j):
	accuracy = 0
	predict = []
	for img in range(len(x[i, j])):  # loop through each image
		if x[i, j, img] == 0:
			predict.append(5)
		else:
			predict.append(9)

	for k in range(len(predict)):
		if predict[k] == y[k]:
			accuracy += 1
	print("Accuracy =", accuracy/len(x[i, j]) * 100, '%')
	return None
def depth2(x, y, x_test, y_test):
	scores_p = scoreFeatures(x, y)
	i_p, j_p = highest_score(scores_p)
	sub_l, sub_r = [], [] 	 		# Assign left dataset and right dataset
	label_l, label_r = [], []		# left and right label

	for img in range(len(x[i_p, j_p])):  # loop through each image
		if x[i_p, j_p, img] == 0:		# pixel=0 go to left
			sub_l.append(x[:, :, img])
			label_l.append(y[img])
		else:		# pixel=1 go to right
			sub_r.append(x[:, :, img])
			label_r.append(y[img])

	# convert to numpy array
	sub_l = np.array(sub_l)
	sub_r = np.array(sub_r)

	# calculate scores for left and right datasets
	scores_l = scoreFeatures(sub_l.transpose(1, 2, 0), label_l)
	scores_r = scoreFeatures(sub_r.transpose(1, 2, 0), label_r)
	i_l, j_l = highest_score(scores_l)
	i_r, j_r = highest_score(scores_r)

	# Prediction
	predict = []
	for img in range(len(x_test[i_p, j_p])):  # loop through each image
		if x_test[i_p, j_p, img] == 0:
			if x_test[i_l, j_l, img] == 0:
				predict.append(5)
			else:
				predict.append(9)
		else:
			if x_test[i_r, j_r, img] == 0:
				predict.append(5)
			else:
				predict.append(9)

	# Calculate accuracy
	accuracy = 0
	for i in range(len(predict)):
		if predict[i] == y_test[i]:
			accuracy += 1
	print("Accuracy =", accuracy / len(x_test[i_p, j_p]) * 100, '%')



def main():
	data = pickle.load(open('data_binarized.pkl', 'rb'))
	scores = scoreFeatures(data['train']['x'], data['train']['y'])
	i, j = highest_score(scores)
	decisiontree_test(data['test']['x'], data['test']['y'], i, j)

	depth2(data['train']['x'], data['train']['y'], data['test']['x'], data['test']['y'])
	#--------------------------------------------------------------------------
	# Your implementation to answer questions on Decision Trees
	#--------------------------------------------------------------------------

	return


if __name__ == "__main__":
	main()
	