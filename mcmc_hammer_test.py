import numpy as np
import emcee
import matplotlib.pyplot as plt
import seaborn as sns

#define the pdf to sample - emcee requires the log(pdf)

def lnprob(x,mu,icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0 #np.dot takes the dot product of two arrays

#Number of dimensions
ndim = 1
means = np.random.rand(ndim)
#Covariance matrix
cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)
#inverse of the covariance matrix
icov = np.linalg.inv(cov)
nwalkers = 250
p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
#Ensemble sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])
#50 burn in steps
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1000)
#--------------------------------------------------------------------------
x = np.linspace(-ndim-0.5,ndim+1,100)
f = np.zeros(len(x))
for ind,i in enumerate(x):
    f[ind] = np.exp(lnprob(i,means[0],icov[0]))
#--------------------------------------------------------------------------
if(1):
    for i in range(ndim):
        plt.figure()
        plt.hist(sampler.flatchain[:,i], 100,normed=True ,color="k", facecolor='blue',label='MCMC')
        plt.plot(x,f,color='k',label='Actual pdf')
        plt.title("Dimension {0:d}".format(i+1))
        plt.legend()
        plt.show()
if(0):
    sns.jointplot(x=sampler.flatchain[:,0], y=sampler.flatchain[:,1], kind="kde",color="g");
    plt.show()

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
