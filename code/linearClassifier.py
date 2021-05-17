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


def softmax(z):
    return 1.0/(1+np.exp(-z))

def linearTrain(x, y):
    #Training parameters
    maxiter = 50
    lamb = 0.01
    eta = 0.01
    
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
    
    class_labels = np.unique(y)
    num_class = class_labels.shape[0]
    assert(num_class == 2) # Binary labels
    num_feats = x.shape[0]
    num_data = x.shape[1]
    
    true_prob = np.zeros(num_data)
    true_prob[y == class_labels[0]] = 1
    
    #Initialize weights randomly
    model = {}
    model['weights'] = np.random.randn(num_feats)*0.01
    # print('w', model['weights'].shape)
    #Batch gradient descent
    verbose_output = False
    for it in range(maxiter):
        prob = softmax(model['weights'].dot(x))
        delta = true_prob - prob
        gradL = delta.dot(x.T)
        model['weights'] = (1 - eta*lamb)*model['weights'] + eta*gradL
    model['classLabels'] = class_labels

    return model


def linearPredict(model, x):
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)

    prob = softmax(model['weights'].dot(x))
    ypred = np.ones(x.shape[1]) * model['classLabels'][1]
    ypred[prob > 0.5] = model['classLabels'][0]

    return ypred


def testLinear():
    #--------------------------------------------------------------------------
    # Your implementation to answer questions on Linear Classifier
    #--------------------------------------------------------------------------
    data = pickle.load(open('data_binarized.pkl', 'rb'))
    x, y = data['train']['x'], data['train']['y']

    # Reshape from HxWx200 to Nx200, N = H*W
    x = np.reshape(x, (len(data['train']['x'][:, :, 1]) * len(data['train']['x'][:, :, 1]),
                       len(data['train']['y'])))
    model = linearTrain(x, y)

    # Reshape testing data
    x_test = data['test']['x']
    x_test = np.reshape(x_test, (len(data['test']['x'][:, :, 1]) * len(data['test']['x'][:, :, 1]),
                       len(data['test']['y'])))
    # Predict test data using trained model
    ypred = linearPredict(model, x_test)
    acc = 0
    for i in range(len(data['test']['y'])):
        # Compare predict to label of testing
        if ypred[i] == data['test']['y'][i]:
            acc += 1
    acc /= len(data['test']['y'])
    print(acc*100, '%')

    # Visualization part
    dimension = model["weights"][:len(model["weights"])-1]
    w = np.reshape(dimension, (int(np.sqrt(len(dimension))), int(np.sqrt(len(dimension)))))
    wp = np.clip(w, 0, None)
    wn = np.clip(w, None, 0)
    plt.subplot(131)
    plt.title('Positive Weights')
    plt.imshow(wp, cmap='gray')
    plt.subplot(133)
    plt.title('Negative Weights')
    plt.imshow(wn, cmap='gray')
    plt.show()

    return None


if __name__ == "__main__":
    testLinear()


