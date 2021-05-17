#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2021
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Mini-Project 6

import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.util import montage


def montageDigits(x,plot=True):
    num_images = x.shape[2]
    m = montage(x.transpose(2, 0, 1), grid_shape=(10, 20))
    if plot:
        plt.imshow(m, cmap='gray')
        plt.axis('off')
        plt.show()

    return np.mean(x, axis=2)


if __name__ == "__main__":
    data = pickle.load(open('data.pkl','rb'))
    avgImg = montageDigits(data['train']['x'])
    plt.imshow(avgImg, cmap='gray')
    plt.axis('off')
    plt.show()

    avg5 = montageDigits(data['train']['x'][:, :, :100], plot=False)
    plt.imshow(avg5, cmap='gray')
    plt.axis('off')
    plt.show()

    avg9 = montageDigits(data['train']['x'][:, :, 100:], plot=False)
    plt.imshow(avg9, cmap='gray')
    plt.axis('off')
    plt.show()


