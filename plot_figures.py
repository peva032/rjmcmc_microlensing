import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import uniform
import matplotlib.mlab as mlab

actual = [0.01,15,10,10,0.001,0.6]

singlechain = pd.read_csv("single_rjmcmc_output_1.csv")
binarychain = pd.read_csv("binary_rjmcmc_output_2.csv")

class uniprior(object):
    def __init__(self,left,right):
        self.left = left
        self.right = right
        self.range = right-left
        self.dist = uniform(left,right)
    def draw(self):
        return rn.uniform(self.left,self.right)
    def pdf(self,x):
        return self.dist.pdf(x)

u0 = uniprior(0,0.1)
t0 = uniprior(1,30)
te = uniprior(1,20)
phi = uniprior(0,20)
q = uniprior(10e-4,0.1)
d = uniprior(0,1)

names = ['u0','t0','te','phi','q','d']
params = {'u0':u0,'t0':t0,'te':te,'phi':phi,'q':q,'d':d}
mu = binarychain.mean()
sigma = binarychain.std()

plt.figure(1)
# binary plots
for i in range(6):
    k = np.linspace(params[names[i]].left,params[names[i]].right,1000)
    plt.subplot(321 + i)
    plt.xlim(params[names[i]].left,params[names[i]].right)
    plt.plot(k,mlab.normpdf(k, mu[i+1], sigma[i+1]))
    plt.vlines(actual[i],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.vlines(mu[i+1],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]),'r')
    plt.ylim(0,1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.xlabel(names[i])

plt.figure(2)
# single lens plots
for i in range(3):
    k = np.linspace(params[names[i]].left,params[names[i]].right,1000)
    plt.subplot(311 + i)
    plt.xlim(params[names[i]].left,params[names[i]].right)
    plt.plot(k,mlab.normpdf(k, mu[i+1], sigma[i+1]))
    plt.vlines(actual[i],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.vlines(mu[i+1],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]),'r')
    plt.ylim(0,1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.xlabel(names[i])

plt.show()
