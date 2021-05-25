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
#==============================================================================
#This code serves the purpose of getting mcmc mh to work for a single lens.
#==============================================================================
#Importing single lens data:

data = pd.read_csv('binary_data.csv')
t = data['t']
y = data['A']

#Initialisation of parameters
param1 = []
param2 = []
param3 = []
param4 = []
param5 = []
param6 = []
theta_store1 = []
theta_store2 = []
theta_store3 = []
theta_store4 = []
theta_store5 = []
theta_store6 = []
theta_acc = []
post_store = []
theta = [rn.uniform(0,3),rn.uniform(10,20),rn.uniform(5,15),rn.uniform(8,12),rn.uniform(0,1),rn.uniform(0,1)]
N =  1000
Nburn = int(N*0.5)
alpha = 0.83
count = 0
m = 2

def u0(u0):
    A0 = [0.4023,0.591,0.0124]
    return (A0[0]*np.exp(-A0[1]*np.log(u0))+A0[2])
def t0(t0):
    DT = [5.558,1.8226]
    return np.exp(-((np.log(t0/0.0202)-DT[0])**2)/(2*DT[1]**2))
def te(te):
    TE = [0.8284,6.0709,1.4283,0.1716,13.7152,1.3578]
    return TE[0]*np.exp(-((np.log(te/0.04)-TE[1])**2)/(2*TE[2]**2)) + TE[3]*np.exp(-((np.log(te/0.04)-TE[4])**2)/(2*TE[5]**2))

# def u0(u0):
#     pu0 = uniform(0,100)
#     return pu0.pdf(u0)
# def t0(t0):
#     pt0 = uniform(1,100)
#     return pt0.pdf(t0)
# def te(te):
#     pte = uniform(1,50)
#     return pte.pdf(te)
def phi(phi):
    pphi = uniform(0,360)
    return pphi.pdf(phi)
def q(q):
    pq = uniform(10e-6,1)
    return pq.pdf(q)
def d(d):
    pd = uniform(0,10.0)
    return pd.pdf(d)

def posterior(t,y,thet):
    return likelihood(m,t,y,thet)*u0(1.0/thet[0])*t0(thet[1])*te(thet[2])*phi(thet[3])*q(thet[4])*d(thet[5])

def prop(th1,th2,alpha):
    cov = (alpha**2)*np.identity(len(th1))
    rv = multivariate_normal.pdf(th1,th2,cov)
    return rv
it = 0
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
for i in bar(range(N)):
    it = it+1
    # theta_prop = theta + np.random.normal(0,alpha,size=6)
    covp = (alpha**2)*np.identity(len(theta))
    theta_prop = multivariate_normal.rvs(theta,covp)
    # print(theta_prop)

    if(all(theta_prop>0)):

        acc = min(1,(posterior(t,y,theta_prop)/posterior(t,y,theta))*(prop(theta,theta_prop,alpha))/prop(theta_prop,theta,alpha))
        u = rn.uniform(0,1)
        if u<=acc:
            # print('Accepted')

            theta = theta_prop
            count = count + 1
            if it>Nburn:
                theta_acc.append(theta)
                post_store.append(posterior(t,y,theta_prop))
                param1.append(theta[0])
                param2.append(theta[1])
                param3.append(theta[2])
                param4.append(theta[3])
                param5.append(theta[4])
                param6.append(theta[5])



p1 = np.mean(np.array(theta_store1[Nburn:])[~np.isnan(theta_store1[Nburn:])])
p2 = np.mean(np.array(theta_store2[Nburn:])[~np.isnan(theta_store2[Nburn:])])
p3 = np.mean(np.array(theta_store3[Nburn:])[~np.isnan(theta_store3[Nburn:])])
p4 = np.mean(np.array(theta_store4[Nburn:])[~np.isnan(theta_store4[Nburn:])])
p5 = np.mean(np.array(theta_store5[Nburn:])[~np.isnan(theta_store5[Nburn:])])
p6 = np.mean(np.array(theta_store6[Nburn:])[~np.isnan(theta_store6[Nburn:])])

print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print("acceptance ratio: ",count/float(N))
t_plot = np.linspace(-80,200,1000)
plt.figure(1)
plt.plot(t,y,'ko',markersize=0.5)
plt.plot(t_plot,binary2(t_plot,p1,p2,p3,p4,p5,p6),'r--',linewidth=1)
plt.axis([0,70,0,12])
plt.show()

plt.figure(1)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store1,label='$u_0$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store2,label='$t_0$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store3,label='$t_e$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store4,label='$phi$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store5,label='$q$',linewidth=0.5)
plt.plot(np.linspace(1,len(theta_store1),len(theta_store1)),theta_store6,label='$s$',linewidth=0.5)
plt.legend()
plt.show()
