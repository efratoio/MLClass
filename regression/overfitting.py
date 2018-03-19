
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(19)

def f(x):

    return np.cos(2 * np.pi * x)


def plot_approximation(est, ax, label=None):

    ax.plot(x_plot, f(x_plot), color='green')
    ax.scatter(X, y, s=10)

    ax.plot(x_plot, est.predict(x_plot[:, np.newaxis]), color='red', label=label)

    ax.set_ylim((-2, 2))

    ax.set_xlim((0, 1))

    ax.set_ylabel('y')

    ax.set_xlabel('x')

    ax.legend(loc='upper right')  #, fontsize='small')



x_plot = np.linspace(0, 1, 100)

n_samples = 20

X = np.random.uniform(0, 1, size=n_samples)[:, np.newaxis]

y = f(X) + np.random.normal(scale=0.4, size=n_samples)[:, np.newaxis]


ax = plt.gca()

ax.plot(x_plot, f(x_plot), color='green')

ax.scatter(X, y, s=10)

ax.set_ylim((-2, 2))

ax.set_xlim((0, 1))

ax.set_ylabel('y')

ax.set_xlabel('x')


fig, axes = plt.subplots(2, 2, figsize=(8, 5))

for ax, degree in zip(axes.ravel(), [0, 2, 4, 8]):

    est = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    est.fit(X, y)

    plot_approximation(est, ax, label='degree=%d' % degree)

plt.show()

