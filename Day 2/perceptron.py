import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron

X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

Y = np.array([0, 1, 1, 0])

clf = Perceptron(tol=1e-3, random_state=23, max_iter=3, penalty='l1')
clf.fit(X,Y)

print(clf.predict([[0,1],[1,0]]))
print(clf.intercept_)
print(clf.coef_)


w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 2)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k-')
plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
plt.show()

