# %%
import scipy.stats
# from typing import Tuple

import math
import numpy as np

import pickle

import os
# import sys

# """ matplotlib drawing to a pdf setup """
# import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#!%matplotlib inline


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # these are needed for deepcopy / pickle
    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)


def save_pdf(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199)


figsize = (6.0, 6.0 * 3 / 4)


def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)


def load_object(filename):
    res = pickle.load(open(filename, "rb"))
    return res


""" Simulation Model, similar to the one in the book 'The Elements of Statistical Learning' """


class G2Model:
    def __init__(self):
        self.K = 3  # mixture components
        self.priors = [0.5, 0.5]
        self.cls = [dotdict(), dotdict()]
        self.cls[0].mus = np.array([[-1, -1], [-1, 0], [0, 0]])
        self.cls[1].mus = np.array([[0, 1], [0, -1], [1, 0]])
        self.Sigma = np.eye(2) * 1 / 20
        self.name = 'GTmodel'

    def samples_from_class(self, c, sample_size):
        """
        :return: x -- [sample_size x d] -- samples from class c
        """
        # draw components
        kk = np.random.randint(0, self.K, size=sample_size)
        x = np.empty((sample_size, 2))
        for k in range(self.K):
            mask = kk == k
            # draw from Gaussian of component k
            x[mask, :] = np.random.multivariate_normal(self.cls[c].mus[k, :], self.Sigma, size=mask.sum())
        return x

    def generate_sample(self, sample_size):
        """
        function to draw labeled samples from the model
        :param sample_size: how many in total
        :return: (x,y) -- features, class, x: [sample_size x d],  y : [sample_size]
        """
        assert (sample_size % 2 == 0), 'use even sample size to obtain equal number of pints for each class'
        y = (np.arange(sample_size) >= sample_size // 2) * 1  # class labels
        x = np.zeros((sample_size, 2))
        for c in [0, 1]:
            # draw from Gaussian Mixture of class c
            x[y == c, :] = self.samples_from_class(c, sample_size // 2)
        y = 2 * y - 1  # remap to -1, 1
        return x, y

    def score_class(self, c, x: np.array) -> np.array:
        """
            Compute log probability for data x and class c (sometimes also called score for the multinomial model)
            x: [N x d]
            return score : [N]
        """
        N = x.shape[0]
        S = np.empty((N, self.K))
        # compute log density of each mixture component
        for k in range(self.K):
            S[:, k] = scipy.stats.multivariate_normal(self.cls[c].mus[k, :], self.Sigma).logpdf(x)
        # compute log density of the mixture
        score = scipy.special.logsumexp(S, axis=1) + math.log(1.0 / self.K) + math.log(self.priors[c])
        return score

    def score(self, x: np.array) -> np.array:
        """ Return log odds (logits) of predictive probability p(y|x) of the network
        """
        scores = [self.score_class(c, x) for c in range(2)]
        score = scores[1] - scores[0]
        return score

    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for a given input
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class -1 or 1 per input point
        """
        return np.sign(self.score(x))

    def test_error(self, predictor, test_data):
        """
        evaluate test error of a predictor
        :param predictor: object with predictor.classify(x:np.array) -> np.array
        :param test_data: tuple (x,y) of the test points
        :return: error rate
        """
        x, y = test_data
        y1 = predictor.classify(x)
        err_rate = (y1 != y).sum() / x.shape[0]
        return err_rate

    def plot_boundary(self, train_data, predictor=None, train_error=None, test_error=None):
        """
        Visualizes the GT model, training points and the decisison boundary of a given predictor
        :param train_data: tuple (x,y)
        predictor: object with
            predictor.score(x:np.array) -> np.array
            predictor.name -- str to appear in the figure
        """
        x, y = train_data
        #
        plt.figure(2, figsize=figsize)
        plt.rc('lines', linewidth=1)
        # plot points
        mask0 = y == -1
        mask1 = y == 1
        plt.plot(x[mask0, 0], x[mask0, 1], 'bo', ms=3)
        plt.plot(x[mask1, 0], x[mask1, 1], 'rd', ms=3)
        # plot classifier boundary
        ngrid = [200, 200]
        xx = [np.linspace(x[:, i].min() - 0.5, x[:, i].max() + 0.5, ngrid[i]) for i in range(2)]
        # xx = [np.linspace(-3, 4, ngrid[i]) for i in range(2)]
        # xx = [np.linspace(-2, 4, ngrid[0]), np.linspace(-3, 3, ngrid[0])]
        Xi, Yi = np.meshgrid(xx[0], xx[1], indexing='ij')  # 200 x 200 matrices
        X = np.stack([Xi.flatten(), Yi.flatten()], axis=1)  # 200*200 x 2
        # Plot the GT scores contour
        score = self.score(X).reshape(ngrid)
        # m1 = np.linspace(0, score.max(), 4)
        # m2 = np.linspace(score.min(), 0, 4)
        # plt.contour(Xi, Yi, score, np.sort(np.concatenate((m1[1:], m2[0:-1]))), linewidths=0.5) # intermediate contour lines of the score
        CS = plt.contour(Xi, Yi, score, [0], colors='r', linestyles='dashed')
        # CS.collections[0].set_label('Bayes optimal')
        # l = dict()
        h, _ = CS.legend_elements()
        H = [h[0]]
        L = ["Bayes optimal"]
        # l[h[0]] = 'GT boundary'
        # CS.collections[0].set_label('GT boundary')
        # Plot Predictor's decision boundary
        if predictor is not None:
            score = predictor.score(X).reshape(ngrid)
            CS = plt.contour(Xi, Yi, score, [0], colors='k', linewidths=1)
            h, _ = CS.legend_elements()
            H += [h[0]]
            L += ["Predictor"]
            # CS.collections[0].set_label('Predictor boundary')
            # h,_ = CS.legend_elements()
            # l[h[0]] = 'GT boundary'
            y1 = predictor.classify(x)
            err = y1 != y
            h = plt.plot(x[err, 0], x[err, 1], 'ko', ms=6, fillstyle='none', label='errors', markeredgewidth=0.5)
            # l[h[0]] = 'Errors'
            H += [h[0]]
            L += ["errors"]
        plt.xlabel("x0")
        plt.ylabel("x1")
        # plt.text(0.3, 1.0, predictor.name, ha='center', va='top', transform=plt.gca().transAxes)
        # plt.legend(loc=0)
        # plt.legend(l.keys(), l.values(), loc=0)
        plt.legend(H, L, loc=0)

        title = f'N={train_data[0].shape[0]}'
        if predictor is not None:
            title += f', D={predictor.hidden_size}'
        if train_error is not None:
            title += f', train error: {train_error*100:.3f}%'
        if test_error is not None:
            title += f', test error: {test_error*100:.3f}%'
        plt.title(title)

# %%


class Lifting:
    def __init__(self, input_size, hidden_size):
        self.W1 = (np.random.rand(hidden_size, input_size) * 2 - 1)
        self.W1 /= np.linalg.norm(self.W1, axis=1).reshape(hidden_size, 1)
        self.b1 = (np.random.rand(hidden_size) * 2 - 1) * 2

    def __call__(self, x):
        """
        input: x [N x 2] data points
        output: [N x hidden_size]
        """
        return np.tanh((x @ self.W1.T + self.b1[np.newaxis, :])*5)


class MyNet:
    """ Template example for the network """

    def __init__(self, input_size, hidden_size):
        # name is needed for printing
        self.hidden_size = hidden_size
        self.name = f'test-net-{hidden_size}'
        self.lifting = Lifting(input_size, hidden_size)

    def score(self, x: np.array) -> np.array:
        """
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: s: np.array [N] predicted scores of class 1 for all points
        """
        psi = self.lifting(x)
        N = x.shape[0]
        psi = np.hstack([psi, np.ones((N, 1))])

        s = psi @ self.theta
        return s

    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for the given gata
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class 0 or 1 per input point
        """
        return np.sign(self.score(x))

    def train(self, train_data: tuple[np.ndarray, np.ndarray], lambda_reg: float = 10e-6):
        """
        Train the model on the provided data
        *
        :param train_data: tuple (x,y) of trianing data arrays: x[N x 2], y[Y]
        """
        x, y = train_data

        # lift the data to D dimensions
        x = self.lifting(x)

        # compute psi matrix which is [N x D+1]
        N = x.shape[0]
        D = self.hidden_size
        psi = np.hstack([x, np.ones((N, 1))])

        # compute weights minimizing MSE
        # (psi^T * psi + lambda * I)^-1 * psi^T * y
        self.theta = np.linalg.inv(psi.T @ psi + lambda_reg * np.eye(D + 1)) @ psi.T @ y


# %%

def experiment(hidden_size: int, train_data_size: int, output_file: str = None) -> tuple[float, float]:
    np.random.seed(seed=1)
    G = G2Model()
    train_data = G.generate_sample(train_data_size)
    test_data = G.generate_sample(50000)
    net = MyNet(2, hidden_size)
    net.train(train_data)
    train_error = G.test_error(net, train_data)
    test_error = G.test_error(net, test_data)

    if output_file is not None:
        G.plot_boundary(train_data, net, train_error=train_error, test_error=test_error)
        plt.draw()
        save_pdf(f'{output_file}.png')
        plt.cla()

    return train_error, test_error

# %%


def boundary():
    folder_name = 'boundary'
    os.makedirs(folder_name, exist_ok=True)

    train_data_size = 40
    Ds = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    for D in Ds:
        train_error, test_error = experiment(D, train_data_size, f'{folder_name}/{D}')
        print(f'Hidden size: {D}')
        print(f'Train error: {train_error*100}%')
        print(f'Test error: {test_error*100}%')

# %%


def variable_training_size():
    np.random.seed(seed=1)
    folder_name = 'training_size'
    os.makedirs(folder_name, exist_ok=True)

    D = 40
    trials = 100
    G = G2Model()

    avg_errors = []
    avg_losses = []

    Ns = list(range(2, 101, 2))
    for N in Ns:
        print(N, flush=True)

        test_errors = []
        test_losses = []
        for _ in range(trials):
            # generate new training data and lifting
            train_data = G.generate_sample(N)
            test_data = G.generate_sample(50000)
            net = MyNet(2, D)

            # train the network
            net.train(train_data)

            # compute loss and error
            test_losses.append(np.mean((net.score(test_data[0]) - test_data[1]) ** 2))
            test_errors.append(G.test_error(net, test_data))

        avg_errors.append(np.mean(np.array(test_errors)))
        avg_losses.append(np.mean(np.array(test_losses)))

    # plot the results
    plt.figure(1, figsize=(10.0, 5.0))

    plt.plot(Ns, avg_errors, marker='d', color='r')
    plt.yscale('log')
    plt.legend(['Test error'], loc=0)
    plt.draw()
    save_pdf(f'{folder_name}/test_error.png')
    plt.cla()

    plt.plot(Ns, avg_losses, marker='d', color='r')
    plt.yscale('log')
    plt.legend(['Test loss'], loc=0)
    plt.draw()
    save_pdf(f'{folder_name}/test_loss.png')
    plt.cla()


# %%
def variable_hidden_size():
    np.random.seed(seed=1)
    folder_name = 'hidden_size'
    os.makedirs(folder_name, exist_ok=True)

    N = 40
    trials = 100
    G = G2Model()

    avg_train_errors = []
    avg_test_errors = []
    avg_train_losses = []
    avg_test_losses = []

    Ds = [1] + list(range(10, 210, 10))
    for D in Ds:
        print(D, flush=True)

        train_errors = []
        test_errors = []
        train_losses = []
        test_losses = []

        for _ in range(trials):

            # generate new training data and lifting
            train_data = G.generate_sample(N)
            test_data = G.generate_sample(50000)
            net = MyNet(2, D)

            # train the network
            net.train(train_data)

            # compute loss and error
            train_losses.append(np.mean((net.score(train_data[0]) - train_data[1]) ** 2))
            test_losses.append(np.mean((net.score(test_data[0]) - test_data[1]) ** 2))
            train_errors.append(G.test_error(net, train_data))
            test_errors.append(G.test_error(net, test_data))

        avg_train_errors.append(np.mean(np.array(train_errors)))
        avg_test_errors.append(np.mean(np.array(test_errors)))
        avg_train_losses.append(np.mean(np.array(train_losses)))
        avg_test_losses.append(np.mean(np.array(test_losses)))

    # plot the results
    plt.figure(1, figsize=(10.0, 5.0))

    plt.plot(Ds, avg_test_errors, marker='d', color='r')
    plt.plot(Ds, avg_train_errors, marker='o', color='b')
    plt.yscale('log')
    plt.legend(['Test error', 'Train error'], loc=0)
    plt.draw()
    save_pdf(f'{folder_name}/errors.png')
    plt.cla()

    plt.plot(Ds, avg_test_losses, marker='d', color='r')
    plt.plot(Ds, avg_train_losses, marker='o', color='b')
    plt.yscale('log')
    plt.legend(['Test loss', 'Train loss'], loc=0)
    plt.draw()
    save_pdf(f'{folder_name}/losses.png')
    plt.cla()


# %%


if __name__ == "__main__":
    boundary()

    variable_training_size()

    variable_hidden_size()
