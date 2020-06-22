import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

np.random.seed(42)
random.seed(42)
sns.set_style("darkgrid")

def generate_linear_data(low, high, m, c, err, num):
    y = list()
    x = np.random.randint(low=low, high=high, size=num)
    x.sort()

    for i in range(num):
        error = random.uniform(0.1, 0.5) * err
        y.append(((m + error) * x[i] )+ c)

    return [x, y]

def plot_reg_line(m, c, x, num):
    y_err = list()
    #Calculate and plot the new estimated best fit
    for j in range(num):
        y_err.append(m * x[j] + c)
    plt.xlim(0, 25)
    plt.ylim(-15, 60)
    plt.plot(np.array(x), np.array(y_err))
    plt.scatter(np.array(x), np.array(y))
    plt.show()
    plt.pause(0.05)
    plt.gcf().clear()

    return sum([abs(y_err[e]-y[e]) for e in range(num)])


num, iterations = 20, 45
x, y = generate_linear_data(low=3, high=20, m=0.9, c=5, err=1.2, num=num)
m, c, learning_rate = random.uniform(-10, 10), random.uniform(-10, 10), 0.002
prev_error = 0
curr_error = 1000

plt.ion()

errors = list()
iters = list()

for i in range(iterations):
    iters.append(i)
    er = plot_reg_line(m, c, x, num)
    errors.append(er)
    prev_error = er
    print(er)
    #Perform gradient descent
    gd_m, gd_c = 0, 0
    for j in range(num):
        gd_m+= ((m * x[j] + c) - y[j]) * x[j]
        gd_c+= (m * x[j] + c) - y[j]

    m = m - (learning_rate/num) * gd_m
    c = c - (learning_rate/num) * gd_c

plt.plot(iters, errors)
print("Number of Iterations: " + str(i+1))
