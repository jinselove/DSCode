import numpy as np
import sys
import os
from lr_utils import load_dataset
import h5py
from PIL import Image
import scipy
from scipy import ndimage
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath("../../../algo/nn"))
from logistic_regression import LogisticRegression
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

if __name__ == '__main__':
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # # Example of a picture
    # index = 25
    # plt.imshow(train_set_x_orig[index])
    # print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    #     "utf-8") + "' picture.")
    # plt.show()

    # show original data shapes
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig[0].shape[0]
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    # Reshape image data to (nx, m) where nx = num_pixels_in_x * num_pixels_in_y
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    # standardize the data set
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # build Logistic regression model
    d = LogisticRegression()

    # train the model
    d.train(train_set_x, train_set_y)
    # d.print_loss_curve()

    # restart the model
    d.train(train_set_x, train_set_y, restart=True, print_cost=True)
    # d.print_loss_curve()

    # predict on the test data
    test_set_pred, test_set_pred_prob = d.predict(test_set_x, test_set_y, print_performance=True)

    # test on an image
    print("*"*40 + "Test on an image file" + "*"*40)
    my_image = "dog.jpg"  # change this to the name of your image file
    fname = "../data/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    x_test = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    y_test_pred, y_test_pred_prob = d.predict(x_test, print_performance=False)
    print(y_test_pred, y_test_pred_prob)
    plt.imshow(image)
    print("y = " + str(np.squeeze(y_test_pred)) + " at a probability of " + str(np.squeeze(y_test_pred_prob)) +
          "; your algorithm predicts a \"" + classes[int(np.squeeze(y_test_pred)),].decode("utf-8") + "\" picture.")
    plt.show()

