
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# In[2]:


# Importing the dataset
data = pd.read_csv('../data/xclara.csv')
print(data.shape)
data.head()


# In[5]:


# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=15)


# In[6]:


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[7]:


# Number of clusters
k = 5
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)


# In[8]:


# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.pause(3)
plt.gcf().clear()


# In[ ]:


def plot(X, clusters, C):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    #plt.show()
    plt.pause(1)
    plt.gcf().clear()


# In[ ]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
#error = dist(C, C_old, None)
error = 0
# Loop will run till the error becomes zero
while error != 10:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    #error = dist(C, C_old, None)
    error+=1
    plot(X, clusters, C)

# In[ ]:




