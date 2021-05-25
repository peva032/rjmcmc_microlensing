import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from binarylc import binary2
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from numpy.linalg import det
from singlelc import MT

def likelihood(m,t,y,thet):
    if m==1:
        sigmae = 0.39
        cov = (sigmae**2)*np.identity(len(y))
        means = MT(t,thet[0],thet[1],thet[2])
        like = multivariate_normal.pdf(y,means,cov)
    if m==2:
        sigmae = 0.39
        cov = (sigmae**2)*np.identity(len(y))
        means = binary2(t,thet[0],thet[1],thet[2],thet[3],thet[4],thet[5])
        like = multivariate_normal.pdf(y,means,cov)
    return like


theta1 = np.linspace(0.2,0.4,100)
data = pd.read_csv('binary_data.csv')
t = data['t']
y = data['A']
like = []

for i in theta1:

    like.append(likelihood(1,t,y,[i,10,10]))

plt.figure()
plt.plot(theta1,like)
plt.show()
