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


def scoreFeatrues(x, y):
	scores = np.zeros(x.shape[:2])
	#--------------------------------------------------------------------------
	# Calculate scores (Implement this)
	#--------------------------------------------------------------------------

	# Calculate each attribute
	for i in range(len(x[:, :, 1])):		# loop through each pixel feature
		for j in range(len(x[i, :, 1])):
			for img in range(len(x[i, j])):		# loop through each image
				if x[i, j, img] == 1:
					if y[img] == 5:
						scores[i, j] += 1
					else:
						scores[i, j] -= 1

	# Calculate score aka attribute
	for i in range(len(x[:, :, 1])):
		for j in range(len(x[i, :, 1])):
			if scores[i, j] > 0:
				scores[i, j] = 1
			else:
				scores[i, j] = 0
	plt.imshow(scores, cmap='gray')
	plt.axis('off')
	plt.show()
	return scores

def highest_score(x, y, scores):

	hi_score = np.zeros(x.shape[:2])
	highest = 0
	r_i, r_j = 0, 0
	for i in range(len(x[:, :, 1])):
		for j in range(len(x[i, :, 1])):
			for img in range(len(x[i, j])):		# loop through each image
				if scores[i, j] == 0 and x[i, j, img] != 0 and y[img] == 9:		# predict 9
					hi_score[i, j] += 1
				if scores[i, j] == 1 and x[i, j, img] != 0 and y[img] == 5: 	# predict 5
					hi_score[i, j] += 1

				check = highest
				highest = max(highest, hi_score[i, j])
				if check != highest:
					r_i, r_j = i, j
	print(r_i, r_j)
	return r_i, r_j


def pred_accuracy(x, y, i, j):
	accuracy = 0
	for img in range(len(x[i, j])):  # loop through each image
		if x[i, j, img] == 1:
			if y[img] == 5:
				accuracy += 1
		else:
			if y[img] == 9:
				accuracy += 1
	print("Accuracy =", accuracy/len(x[i, j]))


def main():
	data = pickle.load(open('data_binarized.pkl', 'rb'))
	scores = scoreFeatrues(data['train']['x'], data['train']['y'])
	i, j = highest_score(data['train']['x'], data['train']['y'], scores)
	pred_accuracy(data['test']['x'], data['test']['y'], i, j)
	#--------------------------------------------------------------------------
	# Your implementation to answer questions on Decision Trees
	#--------------------------------------------------------------------------

	return


if __name__ == "__main__":
	main()
	