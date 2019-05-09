import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the class

        :param learning_rate: learning rate of gradient descent
        :param num_iterations: number of iterations of gradient descent
        :return None
        """
        self.learning_rate = learning_rate
        self.prev_iterations = 0
        self.num_iterations = num_iterations
        self.params = {}
        self.costs = {}
        self.able_to_restart = False

    def sigmoid(self, z):
        """
        Compute the sigmoid of z


        :param z: A scalar or numpy array of any size
        :return s: sigmoid (z)
        """
        s = 1. / (1. + np.exp(-z))

        return s

    def initialize_weights(self, dim):
        """
        This function initialize the weight vector, w, and the bias scalar, b with zeros


        :param dim: size of w vector, i.e., = nx = number of features in a data point
        :return: None
        """
        w = np.zeros((dim, 1))
        b = 0.

        self.params = {"w": w, "b": b}

    def propagate(self, w, b, X, Y):
        """
        Calculate the cost function, J, in a forward propagate step and the gradient of J over w and b in a back
        propagate step.

        :param w: weights, size = (nx, 1)
        :param b: bias, scalar
        :param X: data matrix, size = (nx, m), where m is the total number of sample data points
        :param Y: label vector, size = (1, m), the elements are binary
        :return grads: a dictionary, {"dw: dw, "db":db}, that contains the gradient of the loss
                       with respect to w and b
        :return cost: average loss of all samples
        """
        # forward propagate
        m = X.shape[1]
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        loss = Y * np.log(A) + (1. - Y) * np.log(1. - A)
        cost = (-1. / m) * np.sum(loss)

        # backward propagate
        dZ = A - Y
        dw = 1. / m * np.dot(X, dZ.T)
        db = 1. / m * np.sum(dZ)
        grads = {"dw": dw, "db": db}

        return grads, cost

    def optimize(self, X, Y, print_cost=True):
        """
        Optimize self.params, i.e., w and b, using gradient descent algorithm

        :param X: data X, size = (nx, m)
        :param Y: data labels, size = (1, m)
        :param print_cost: print total cost during iterations if true
        """
        w = self.params["w"]
        b = self.params["b"]
        for i in range(self.prev_iterations, self.num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            if i % 100 == 0:
                self.costs[i] = cost
                if print_cost:
                    print("Iteration {0}, cost = {1}".format(i, cost))

        # renew self.params using updated w and b
        self.params["w"] = w
        self.params["b"] = b

    def train(self, X_train, Y_train, restart=False, num_iterations=None, print_cost=True):
        """
        Train the model from zero coefficients and bias

        :param X_train: training data X, shape = (nx, m)
        :param Y_train: training data Y, shape = (1, m)
        :param restart: if true, the model will train from lasted saved state, i.e., self.params and self.costs
        :param num_iterations: only useful when restart is True
        :param print_cost: print total cost during iterations if true
        :return: None
        """
        if restart:
            if not self.able_to_restart:
                print("Error: cannot restart from a non-trained model. Exiting ...")
                sys.exit(-1)
            if not num_iterations:
                num_iterations = self.num_iterations
            self.prev_iterations = self.num_iterations
            self.num_iterations += num_iterations
        else:
            self.initialize_weights(X_train.shape[0])

        self.optimize(X_train, Y_train, print_cost)
        self.able_to_restart = True

    def predict(self, X_test, Y_test=None, threshold=0.5, print_performance=True):
        """

        :param X_test: test data X, shape = (nx, m_test)
        :param Y_test: test data Y, shape = (1, m_test)
        :param threshold: the probability threshold to transform probabilities to a binary outputs
        :param print_performance: print the performance of the classification model if True
        :return Y_predictions: a numpy array containing binary predictions for the data points in X_test
        """
        # check X_test data shape
        if self.params["w"].shape[0] != X_test.shape[0]:
            print("Error: shape[0] of test data, i.e., number of features is not equal to that of the training data. "
                  "Exiting ...")
            sys.exit(-1)

        m = X_test.shape[1]
        Z_predictions = np.dot(self.params["w"].T, X_test) + self.params["b"]
        Y_predictions_prob = self.sigmoid(Z_predictions)
        Y_predictions = (Y_predictions_prob >= threshold) * 1.0

        assert(Y_predictions_prob.shape == (1, m))
        assert(Y_predictions.shape == (1, m))

        if print_performance and Y_test is not None:
            self.print_model_performance(np.squeeze(Y_test), np.squeeze(Y_predictions), np.squeeze(Y_predictions_prob))

        return Y_predictions, Y_predictions_prob

    def print_loss_curve(self):
        """
        Draw the curve of cost as a function of iterations (per hundreds)

        :return: None
        """
        plt.figure()
        plt.plot(self.costs.keys(), self.costs.values())
        plt.show()

    @staticmethod
    def print_model_performance(Y_test, Y_test_pred, Y_test_pred_prob):
        """
        Print out the model performance

        :param Y_test: Y of test data, shape = (1, m)
        :param Y_test_pred: binary prediction test data, shape = (1, m)
        :param Y_test_pred_prob: prediction probability of test data, shape = (1, m)
        :return: None
        """
        print("-"*40 + "Accuracy" + "-"*40)
        print(accuracy_score(Y_test, Y_test_pred))

        print("-" * 40 + "Recall of positive class" + "-" * 40)
        print(recall_score(Y_test, Y_test_pred))

        print("-" * 40 + "Precision of positive class" + "-" * 40)
        print(precision_score(Y_test, Y_test_pred))

        print("-" * 40 + "ROC AUC of Score" + "-" * 40)
        print(roc_auc_score(Y_test, Y_test_pred_prob))






