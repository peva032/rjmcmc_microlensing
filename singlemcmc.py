import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from singlelc import MT
from binarylc import binary2
from scipy.stats import uniform
import math
import progressbar
import numpy.random as rn
from scipy import stats
from scipy.optimize import leastsq
from scipy.stats import multivariate_normal
from likelihood import likelihood
from scipy.stats import mvn
#==============================================================================
#This code serves the purpose of getting mcmc mh to work for a single lens.
#==============================================================================
#Importing single lens data:

data = pd.read_csv('single_data.csv')
t = data['t']
y = data['A']

#Initialisation of parameters
param1 = []
param2 = []
param3 = []
theta = [rn.uniform(0,4),rn.uniform(5,20),rn.uniform(1,20)]
N = 10000
Nburn = N/2
#sigmae = 0.01 #additive noise of the simulated data
alpha = 0.9
count = 0
theta_store1 = []
theta_store2 = []
theta_store3 = []
m = 1

def u0(u0):
    pu0 = uniform(0,100)
    return pu0.pdf(u0)
def t0(t0):
    pt0 = uniform(1,100)
    return pt0.pdf(t0)
def te(te):
    pte = uniform(1,50)
    return pte.pdf(te)

def posterior(t,y,thet):
    return likelihood(m,t,y,thet)*u0(thet[0])*t0(thet[1])*te(thet[2])

def prop(th1,th2,alpha):
    cov = (alpha**2)*np.identity(len(th1))
    p = multivariate_normal.pdf(th1,th2,cov)
    return p


it = 0
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
for i in bar(range(N)):
    it = it+1
    covp = (alpha**2)*np.identity(len(theta))
    theta_prop = multivariate_normal.rvs(theta,covp)

    print(theta_prop)
    if(all(theta_prop>0)):
        # acc_m = stats.norm.cdf(theta[0])*stats.norm.cdf(theta[1])*stats.norm.cdf(theta[2])\
        # /stats.norm.cdf(theta_prop[0])*stats.norm.cdf(theta_prop[1])*stats.norm.cdf(theta_prop[2])
        acc = min(1,(posterior(t,y,theta_prop)/posterior(t,y,theta))*prop(theta,theta_prop,alpha)/prop(theta_prop,theta,alpha))
        u = rn.uniform(0,1)
        print(u,acc)
        if u<=acc:
            print('Accepted')
            theta = theta_prop
            count = count+1
            if it>Nburn:
                param1.append(theta[0])
                param2.append(theta[1])
                param3.append(theta[2])

                theta_store1.append(theta[0])
                theta_store2.append(theta[1])
                theta_store3.append(theta[2])

print(np.mean(param1))
print(np.mean(param2))
print(np.mean(param3))
print("acceptance ratio: ",count/float(N))
t_plot = np.linspace(0,60,100)
plt.figure()
plt.plot(t,y,'ko',markersize=0.7)
plt.plot(t_plot,MT(t_plot,np.mean(param1),np.mean(param2),np.mean(param3)),'r--',linewidth=0.5)
plt.show()

plt.figure()
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store1,label='$u_0$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store2,label='$t_0$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store3,label='$t_e$',linewidth=0.5)
plt.legend()
plt.show()
