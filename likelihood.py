import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from binarylc import binary2
from scipy.stats import multivariate_normal
from singlelc import MT

def likelihood(m,t,y,thet):
    if m==1:
        # sigmae = 0.4
        # cov = cov or (sigmae**2)*np.identity(len(y))
        means = MT(t,thet[0],thet[1],thet[2])
        like = np.exp(-np.linalg.norm(y-means)/float(2/0.4/0.4))#unnormalized! Fine in this case only
    if m==2:
        # sigmae = 0.4
        # cov = cov or (sigmae**2)*np.identity(len(y))
        means = binary2(t,thet[0],thet[1],thet[2],thet[3],thet[4],thet[5])
        like = np.exp(-np.linalg.norm(y-means)/float(2/0.4/0.4))#multivariate_normal.pdf(y,means,cov)
    return like

if(0):
    theta1 = np.linspace(0,0.02,100)
    data = pd.read_csv('binary_data.csv')
    t = data['t']
    y = data['A']
    like = []

    for i  in theta1:

        like.append(likelihood(2,t,y,[i,15,10,10,0.001,0.6]))

    plt.figure()
    plt.plot(theta1,like)
    plt.show()


    theta2 = np.linspace(0,0.6,100)
    data = pd.read_csv('single_data.csv')
    t2 = data['t']
    y2 = data['A']
    like2 = []

    for ii in theta2:

        like2.append(likelihood(1,t2,y2,[ii,10,10]))

    plt.figure()
    plt.plot(theta2,like2)
    plt.show()
