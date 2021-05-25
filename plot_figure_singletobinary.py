from rjfun import *
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab
from functools import reduce
from operator import add

actual=[(0.01,15,10,0.17453,0.001,d,0.4) for d in np.linspace(0.6,1.0,3)]

params = ['u0','t0','te','phi','q','d']
u0 = betaprior(1,15)
t0 = normprior(12,7)
te = normprior(12,7)
phi = uniprior(0,2*np.pi)
q = uniprior(0,1)
d = uniprior(0,1)
priors = {'u0':u0,'t0':t0,'te':te,'phi':phi,'q':q,'d':d}
# ii=1
# p=actual[ii]


for ii,p in enumerate(actual):

    singlechain = pd.read_csv("outputs/singletobinary/singletobinary_%d_output_(single).csv" %(ii))
    binarychain = pd.read_csv("outputs/singletobinary/singletobinary_%d_output_(binary).csv" %(ii))
    mu = binarychain.mean()
    sigma = binarychain.std()
    binpredict = tuple([mu[par] for par in params])
    mu = singlechain.mean()
    sigma = singlechain.std()
    sinpredict = tuple([mu[par] for par in params[0:3]])

    parstring = reduce(add,[str("_%.3f" % par) for par in p])
    data=pd.read_csv("outputs/singletobinary/singletobinary_data%s.csv" %(parstring))
    t = np.array(data['t'])
    A = np.array(data['A'])
    N = len(A)
    AA = binary2(t,*p[:-1])
    tt = np.linspace(10,20,N)
    plt.figure(1)
    # plt.plot(t,A,'kx',markersize=0.8)
    plt.plot(tt,binary2(tt,*p[:-1]),'b--',linewidth=0.5)

    plt.figure(2)
    plt.subplot(320+(ii+1)*2-1)
    plt.plot(t,A,'kx',markersize=0.8)
    plt.plot(tt,MT(tt,*sinpredict),'g--')
    plt.plot(tt,binary2(tt,*binpredict),'b--')

    count1 = singlechain.shape[0]
    count2 = binarychain.shape[0]
    p1 = count1/float(count1+count2)
    p2 = count2/float(count1+count2)
    var = 2*np.sqrt((1-p1)*p1/float(count1+count2))

    plt.subplot(320+(ii+1)*2)
    plt.scatter([1.,2.],[p1,p2])
    plt.xticks([1,2],["Single Lens","Binary Lens"])
    plt.xlim([0.5,2.5])
    plt.errorbar([1.,2.],[p1,p2],yerr=[var,var],linestyle='None')
    plt.ylim([0,1])
    plt.title('Probability of Models')
    plt.ylabel('Predicted Probability')
    plt.xlabel('$Model$')


plt.show()
