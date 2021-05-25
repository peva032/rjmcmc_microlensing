from scipy.stats import multivariate_normal,uniform,gamma,beta,norm
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

mu = np.array([1,2])
sig = np.array([1,3])
cov = np.diag(sig**2)
N = 100
k = np.linspace(-10,10,N)
pot = np.zeros([N,N])
X = np.zeros([N,N])
Y = np.zeros([N,N])

for i,j in product(range(N),range(N)):
    X[i][j]=k[i]
    Y[i][j]=k[j]
    pot[i][j] = multivariate_normal.pdf(x=[X[i][j],Y[i][j]], mean=mu, cov=cov)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X,Y,pot, rstride=1, cstride=1)
plt.plot(0*k+10,k,[norm(mu[0],sig[0]).pdf(k[]) for i in range(N)])
plt.show()
