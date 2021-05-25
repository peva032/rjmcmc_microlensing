import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import uniform
import matplotlib.mlab as mlab

# actual = [0.01,15,10,10,0.001,0.6]

singlechain = pd.read_csv("binary_real_rjmcmc_output_1.csv")
binarychain = pd.read_csv("binary_real_rjmcmc_output_2.csv")

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

u0 = uniprior(0,1)
t0 = uniprior(400,600)
te = uniprior(10,100)
phi = uniprior(0,360)
d = uniprior(0,1)
q = uniprior(10e-4,0.1)

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
    # plt.vlines(actual[i],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
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
    # plt.vlines(actual[i],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.vlines(mu[i+1],[0],1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]),'r')
    plt.ylim(0,1.2*mlab.normpdf(mu[i+1],mu[i+1],sigma[i+1]))
    plt.xlabel(names[i])

plt.figure(3)
plt.subplot(311)
plt.xlabel('u0')
hist, bins = np.histogram(singlechain['u0'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[0],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(u0.left,u0.right)
plt.subplot(312)
plt.xlabel('t0')
hist, bins = np.histogram(singlechain['t0'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[1],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(t0.left,t0.right)
plt.subplot(313)
plt.xlabel('te')
hist, bins = np.histogram(singlechain['te'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[2],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(te.left,te.right)

plt.figure(4)
plt.subplot(321)
plt.xlabel('u0')
hist, bins = np.histogram(binarychain['u0'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[0],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(u0.left,u0.right)
plt.subplot(322)
plt.xlabel('t0')
hist, bins = np.histogram(binarychain['t0'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[1],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(t0.left,t0.right)
plt.subplot(323)
plt.xlabel('te')
hist, bins = np.histogram(binarychain['te'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[2],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(te.left,te.right)
plt.subplot(324)
plt.xlabel('phi')
hist, bins = np.histogram(binarychain['phi'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[3],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(phi.left,phi.right)
plt.subplot(325)
plt.xlabel('q')
hist, bins = np.histogram(binarychain['q'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[4],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(q.left,q.right)
plt.subplot(326)
plt.xlabel('d')
hist, bins = np.histogram(binarychain['d'], bins=15, density=True)
widths = np.diff(bins)
plt.bar(bins[:-1], hist, widths)
# plt.vlines(actualparams[5],[0],[100],'r')
plt.ylim((0,1.2*max(hist)))
plt.xlim(d.left,d.right)

count1 = singlechain.shape[0]
count2 = binarychain.shape[0]
p1 = count1/float(count1+count2)
p2 = count2/float(count1+count2)
e = 2*np.sqrt((1-p1)*p1/float(count1+count2))

plt.figure(5)
plt.scatter([1.,2.],[p1,p2])
plt.xticks([1,2],["Single Lens","Binary Lens"])
plt.xlim([0.5,2.5])
plt.errorbar([1.,2.],[p1,p2],yerr=[e,e],linestyle='None')
plt.ylim([0,1])
plt.title('Probability of Models')
plt.ylabel('Predicted Probability')
plt.xlabel('$Model$')

# plt.plot(data['t'],data['A'],'ko',label='data',markersize=0.8)
# plt.plot(t_data,MT(t_data,theta_op1[0],theta_op1[1],theta_op1[2]),'r--',label='single lens model',linewidth=0.5)
# plt.plot(t_data,binary2(t_data,theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5]),'b--',label='binary lens model',linewidth=0.5)
# # plt.axis([-10,100,0,13])
# plt.title('$Simulated\; data\; with\; rjmcmc\; model\; estimates$')
# plt.legend()
# plt.xlabel('$t$')
# plt.ylabel('$A(t)$')


plt.show()
