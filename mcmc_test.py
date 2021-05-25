
"""One Dimensional MCMC - Metropolis Hastings Example"""

import numpy as np
import scipy
import seaborn as sbn
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#---------------------------------------------------------------------------
#Initial starting point in parameter space
#---------------------------------------------------------------------------

def post(x):
    p = (1/(2*np.sqrt(1.5*np.pi)))*np.exp(-((x-4)**2)/(2*1.5**2)) + (1/(0.5*np.sqrt(2*np.pi)))*np.exp(-((x-7)**2)/(2*0.5**2))
    return p

N = 3000000
s = 10
r = 0
p = post(r)
alpha = 0.9

samples = []

for i in range(N):
    rn = r + alpha*np.random.normal()
    pn = post(rn)
    if pn >= p:
        p = pn
        r = rn
    else:
        u = np.random.rand()
        if u < pn/p:
            p = pn
            r = rn
    if i % s == 0:
        samples.append(r)

samples = np.array(samples)


        #samples.append(theta)

x = np.linspace(-4,12,1000)
plt.figure(1)
plt.plot(x,post(x))
plt.hist(samples, bins=100,normed = True, color='grey',label='Density')
plt.title('Metropolis-Hastings')
plt.show()
