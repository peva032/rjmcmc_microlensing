import matplotlib.pyplot as plt
from rjfun import binary2
import numpy as np
import pandas as pd
import math
import numpy.random as rn

#This script is designed to creat binary_data.csv which is used as simulation
# data for rjmcmc_single_binary.py
#-----------------------------------------------------------------------------
def generatebinary(u0,t0,te,phi,q,d,sigmae):
    # u0 = 0.01
    # t0 = 15
    # te = 10
    # phi = np.pi/2
    # q = 0.1
    # d = 0.6

    # np.random.seed(1)
    #generating zs path
    A = []
    # sigmae = 0.3
    N = 2000
    T  = 40
    e = rn.normal(0,sigmae,N)

    t = rn.uniform(0,T,N)

    A = binary2(t,u0,t0,te,phi,q,d)

    AA = A
    A = A+e
    tt = np.linspace(0,T,N)


    if(0):
        plt.figure(1)
        plt.plot(t,A,'kx',markersize=0.8)
        plt.plot(tt,binary2(tt,u0,t0,te,phi,q,d),'b--',linewidth=0.5)
        plt.show()
        plt.figure(2)
        plt.plot(xs_binary,ys_binary,label='source path')
        plt.plot(z2,0,'yo',markersize=12,label='m1')
        plt.plot(z1,0,'ko',markersize=8,label='m2')
        plt.legend()
        plt.xlabel('$x_s$')
        plt.ylabel('$y_s$')
        plt.title('$Source\; path\; in\; lens\; plane$')
        plt.grid()
        plt.show()


    data = {'t':t,'A':A}
    binary_data = pd.DataFrame(data)
    binary_data.to_csv('binary_data.csv',index=False)
