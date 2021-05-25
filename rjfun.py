import time
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
import VBBinaryLensingLibrary as vbb
from scipy.stats import multivariate_normal,uniform,gamma,beta,norm
from functools import reduce
from operator import mul
from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

# Define Prior Class
class uniprior(object):
    def __init__(self,left,right):
        self.left = left
        self.right = right
        self.range = (right-left)/2.0
        self.dist = uniform(left,right)
        self.isnorm = False
    def draw(self):
        return rn.uniform(self.left,self.right)
    def pdf(self,x):
        return self.dist.pdf(x)

class betaprior(object):
    def __init__(self,a,b):
        self.alpha = a
        self.beta = b
        self.left = 0
        self.right = 1
        self.range = self.right - self.left
        self.dist = beta(a,b)
        self.isnorm = False
    def draw(self):
        return rn.beta(self.alpha,self.beta)
    def pdf(self,x):
        return self.dist.pdf(x)
    def logpdf(self,x):
        return self.dist.logpdf(x)

class gammaprior(object):
    def __init__(self,sh,sc):
        self.shape = float(sh)
        self.scale = float(sc)
        self.dist = gamma(sh,loc=0,scale=sc)
        self.left = 0
        self.range = self.dist.std
        self.right = self.left + 2*self.range
        self.isnorm = False
    def draw(self):
        return rn.gamma(self.shape,self.scale)
    def pdf(self,x):
        return self.dist.pdf(x)

class normprior(object):
    def __init__(self,mu,sig):
        self.mu = mu
        self.sigma = sig
        self.range = sig
        self.left = mu - 3*sig
        self.right = mu + 3*sig
        self.dist = norm(mu,sig)
        self.isnorm = True
    def draw(self):
        return self.mu + (self.sigma)*rn.randn()
    def pdf(self,x):
        return self.dist.pdf(x)
    def logpdf(self,x):
        return self.dist.logpdf(x)

def offsupport(m,theta,priors,params):
    if (m==1):
        return (reduce(mul,[priors[p].pdf(theta[i]) for i,p in enumerate(params[0:3])])==0.0)
    else:
        return (reduce(mul,[priors[p].pdf(theta[i]) for i,p in enumerate(params)])==0.0)

def proposal(theta,covp,priors,params):
    return multivariate_normal.rvs(mean=theta, cov=covp)

def proposal2(m,m_prop,theta,covp,priors,params):
    if (m==1 and m_prop==1):
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp[0:3,0:3])
        # while reduce(mul,[priors[p].pdf(theta_prop[i]) for i,p in enumerate(params[0:3])])==0.0:
        #     theta_prop = multivariate_normal.rvs(mean=theta, cov=covp[0:3,0:3])
        return theta_prop
    elif (m==1 and m_prop==2):
        # return np.append(theta,[priors[p].draw() for p in params[3:6]])
        theta = np.append(theta,np.zeros(np.shape(theta)))
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
        # while reduce(mul,[priors[p].pdf(theta_prop[i]) for i,p in enumerate(params)])==0.0:
        #     theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
        return theta_prop
    elif (m==2 and m_prop==1):
        # return theta[0:3]
        theta_prop = multivariate_normal.rvs(mean=theta[0:3], cov=covp[0:3,0:3])
        # while reduce(mul,[priors[p].pdf(theta_prop[i]) for i,p in enumerate(params[0:3])])==0.0:
        #     theta_prop = multivariate_normal.rvs(mean=theta[0:3], cov=covp[0:3,0:3])
        return theta_prop
    elif (m==2 and m_prop==2):
        theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
        # while reduce(mul,[priors[p].pdf(theta_prop[i]) for i,p in enumerate(params)])==0.0:
        #     theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
        return theta_prop

#==============================================================================
#Defining poseterior

def post(m,t,y,theta,cov,priors):
    if m==1:
        return likelihood(m,t,y,theta,cov)*priors['u0'].pdf(theta[0])*priors['t0'].pdf(theta[1])*priors['te'].pdf(theta[2])
    if m==2:
        return likelihood(m,t,y,theta,cov)*priors['u0'].pdf(theta[0])*priors['t0'].pdf(theta[1])*priors['te'].pdf(theta[2])*priors['phi'].pdf(theta[3])*priors['q'].pdf(theta[4])*priors['d'].pdf(theta[5])


def prior_ratio(m,m_prop,theta,theta_prop,priors):
    # names=['u0','t0','te','phi','q','d']
    # for i,n in enumerate(names):
    #     print("%s-ratio: %f" % (n,priors[n].pdf(theta_prop[i])/priors[n].pdf(theta[i])))
    if (m==1 and m_prop==1):
        #print('Calculating prior ratio 1->1')
        return np.exp(priors['u0'].logpdf(theta_prop[0])+priors['t0'].logpdf(theta_prop[1])+priors['te'].logpdf(theta_prop[2]) \
        -priors['u0'].logpdf(theta[0])-priors['t0'].logpdf(theta[1])-priors['te'].logpdf(theta[2]))
        # return priors['u0'].pdf(theta_prop[0])\
        # *priors['t0'].pdf(theta_prop[1])\
        # *priors['te'].pdf(theta_prop[2])\
        # /(priors['u0'].pdf(theta[0])\
        # *priors['t0'].pdf(theta[1])\
        # *priors['te'].pdf(theta[2]))
    # elif(m==1 and m_prop==2):
    #     return np.exp(priors['u0'].logpdf(theta_prop[0])+priors['t0'].logpdf(theta_prop[1])+priors['te'].logpdf(theta_prop[2]) \
    #     -priors['u0'].logpdf(theta[0])-priors['t0'].logpdf(theta[1])-priors['te'].logpdf(theta[2]))\
    #     *(priors['phi'].pdf(theta_prop[3])\
    #     *priors['q'].pdf(theta_prop[4])\
    #     *priors['d'].pdf(theta_prop[5]))
    # elif(m==2 and m_prop==1):
    #     return np.exp(priors['u0'].logpdf(theta_prop[0])+priors['t0'].logpdf(theta_prop[1])+priors['te'].logpdf(theta_prop[2]) \
    #     -priors['u0'].logpdf(theta[0])-priors['t0'].logpdf(theta[1])-priors['te'].logpdf(theta[2]))\
    #     /(priors['phi'].pdf(theta[3])\
    #     *priors['q'].pdf(theta[4])\
    #     *priors['d'].pdf(theta[5]))
    else:
        #print('Calculating prior ratio 1->2')
        return np.exp(priors['u0'].logpdf(theta_prop[0])+priors['t0'].logpdf(theta_prop[1])+priors['te'].logpdf(theta_prop[2]) \
        -priors['u0'].logpdf(theta[0])-priors['t0'].logpdf(theta[1])-priors['te'].logpdf(theta[2]))\
        *priors['phi'].pdf(theta_prop[3])\
        *priors['q'].pdf(theta_prop[4])\
        *priors['d'].pdf(theta_prop[5])\
        /(priors['phi'].pdf(theta[3])\
        *priors['q'].pdf(theta[4])\
        *priors['d'].pdf(theta[5]))
        # return priors['u0'].pdf(theta_prop[0])\
        # *priors['t0'].pdf(theta_prop[1])\
        # *priors['te'].pdf(theta_prop[2])\
        # *priors['phi'].pdf(theta_prop[3])\
        # *priors['q'].pdf(theta_prop[4])\
        # *priors['d'].pdf(theta_prop[5])\
        # /(priors['u0'].pdf(theta[0])\
        # *priors['t0'].pdf(theta[1])\
        # *priors['te'].pdf(theta[2])\
        # *priors['phi'].pdf(theta[3])\
        # *priors['q'].pdf(theta[4])\
        # *priors['d'].pdf(theta[5]))

def prior_ratio2(m,m_prop,theta,theta_prop,priors):
    # names=['u0','t0','te','phi','q','d']
    # for i,n in enumerate(names):
    #     print("%s-ratio: %f" % (n,priors[n].pdf(theta_prop[i])/priors[n].pdf(theta[i])))
    if (m==1 and m_prop==1):
        return priors['u0'].pdf(theta_prop[0])\
        *priors['t0'].pdf(theta_prop[1])\
        *priors['te'].pdf(theta_prop[2])\
        /(priors['u0'].pdf(theta[0])\
        *priors['t0'].pdf(theta[1])\
        *priors['te'].pdf(theta[2]))
    elif (m==2 and m_prop==1):
        return priors['u0'].pdf(theta_prop[0])\
        *priors['t0'].pdf(theta_prop[1])\
        *priors['te'].pdf(theta_prop[2])\
        /(priors['u0'].pdf(theta[0])\
        *priors['t0'].pdf(theta[1])\
        *priors['te'].pdf(theta[2])\
        *priors['phi'].pdf(theta[3])\
        *priors['q'].pdf(theta[4])\
        *priors['d'].pdf(theta[5]))
    elif (m==1 and m_prop==2):
        return priors['u0'].pdf(theta_prop[0])\
        *priors['t0'].pdf(theta_prop[1])\
        *priors['te'].pdf(theta_prop[2])\
        *priors['phi'].pdf(theta_prop[3])\
        *priors['q'].pdf(theta_prop[4])\
        *priors['d'].pdf(theta_prop[5])\
        /(priors['u0'].pdf(theta[0])\
        *priors['t0'].pdf(theta[1])\
        *priors['te'].pdf(theta[2]))
    else:
        return priors['u0'].pdf(theta_prop[0])\
        *priors['t0'].pdf(theta_prop[1])\
        *priors['te'].pdf(theta_prop[2])\
        *priors['phi'].pdf(theta_prop[3])\
        *priors['q'].pdf(theta_prop[4])\
        *priors['d'].pdf(theta_prop[5])\
        /(priors['u0'].pdf(theta[0])\
        *priors['t0'].pdf(theta[1])\
        *priors['te'].pdf(theta[2])\
        *priors['phi'].pdf(theta[3])\
        *priors['q'].pdf(theta[4])\
        *priors['d'].pdf(theta[5]))

#==============================================================================
#Calculating posterior ratio:

def post_ratio(t,y,m,m_prop,theta,theta_prop,cov,priors):
    return post(m_prop,t,y,theta_prop,cov,priors)/post(m,t,y,theta,cov,priors)

#==============================================================================
#Calculating posterior ratio:

@timeout(10)
def likelihood_ratio(t,y,m,m_prop,theta,theta_prop,cov):
    # start_time = time.time()
    #print('Calculating lh ratio')
    # if m==1:
    #     means = MT(t,theta[0],theta[1],theta[2])
    # elif m==2:
    #     means = binary2(t,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5])
    # if m_prop==1:
    #     means_prop = MT(t,theta_prop[0],theta_prop[1],theta_prop[2])
    # elif m_prop==2:
    #     means_prop = binary2(t,theta_prop[0],theta_prop[1],theta_prop[2],theta_prop[3],theta_prop[4],theta_prop[5])
    # return np.exp(-(potential(y,means_prop,cov)-potential(y,means,cov))/2)
    # return likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov)
    lhratio = np.exp(loglikelihood(m_prop,t,y,theta_prop,cov)-loglikelihood(m,t,y,theta,cov))
    # print(time.time()-start_time)
    return lhratio
#==============================================================================
#Defining likelihood functions
def loglikelihood(m,t,y,theta,cov):
    # covi = [1/cov[i][i] for i in np.shape(cov)[0]]
    if m==1:
        #print('getting Ax for model 1')
        means = MT(t,theta[0],theta[1],theta[2])
        # Y=np.reshape(y-means,(1,len(y)))
        # like = multivariate_normal.pdf(np.zeros(np.shape(means)),mean=y-means,cov=cov)
        # like = np.exp(-(0.5)*np.dot(Y,np.dot(np.linalg.inv(cov),Y.transpose())))
        # like = np.exp(-sum([(y[i] - means[i])^2/cov[i][i] for i in range(len(means))])/2)#unnormalized! Fine in this case only
    if m==2:
        #print('getting Ax for model 2 phi=%.3f,q=%.3f,d=%.3f'%(theta[3],theta[4],theta[5]))
        # means = MT(t,thet[0],thet[1],thet[2])
        means = binary2(t,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5])
        # Y=np.reshape(y-means,(1,len(y)))
        # like = np.exp(-(0.5)*np.dot(Y,np.dot(np.linalg.inv(cov),Y.transpose())))#multivariate_normal.pdf(y,means,cov)
    # X=(y-means)
    # like = np.exp(-np.dot(X.transpose(),np.dot(np.linalg.pinv(cov),X))/2)
    #print('calculating likelihood')
    like = multivariate_normal.logpdf(np.zeros(np.shape(means)),mean=y-means,cov=cov)
    return like

def likelihood(m,t,y,theta,cov):
    # covi = [1/cov[i][i] for i in np.shape(cov)[0]]
    if m==1:
        # print('calc m1')
        means = MT(t,theta[0],theta[1],theta[2])
        # Y=np.reshape(y-means,(1,len(y)))
        # like = multivariate_normal.pdf(np.zeros(np.shape(means)),mean=y-means,cov=cov)
        # like = np.exp(-(0.5)*np.dot(Y,np.dot(np.linalg.inv(cov),Y.transpose())))
        # like = np.exp(-sum([(y[i] - means[i])^2/cov[i][i] for i in range(len(means))])/2)#unnormalized! Fine in this case only
    if m==2:
        # print('calc m1')
        # means = MT(t,thet[0],thet[1],thet[2])
        means = binary2(t,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5])
        # Y=np.reshape(y-means,(1,len(y)))
        # like = multivariate_normal.pdf(np.zeros(np.shape(means)),mean=y-means,cov=cov)
        # like = np.exp(-(0.5)*np.dot(Y,np.dot(np.linalg.inv(cov),Y.transpose())))#multivariate_normal.pdf(y,means,cov)
    X=(y-means)
    like = np.exp(-np.dot(X.transpose(),np.dot(np.linalg.pinv(cov),X))/2)
    return like

def potential(y,mean,cov):
    return np.dot((y-mean).transpose(),np.dot(np.linalg.inv(cov),y-mean))

#==============================================================================
#Calculating the correction factor with proposal distribution:
#Since our proposal dist for the intermodal jump is uniform we will use draw from a unifrom
#distribution for our correction factor

def prop_ratio(m,m_prop,theta,theta_prop,priors):
    if(m_prop==2 and m==1):
        return 1.0/priors['phi'].pdf(theta[3])*priors['q'].pdf(theta[4])*priors['d'].pdf(theta[5])
    elif(m_prop==1 and m==2):
        return priors['phi'].pdf(theta_prop[3])*priors['q'].pdf(theta_prop[4])*priors['d'].pdf(theta_prop[5])
    elif(m_prop==m and m==1):
        return priors['phi'].pdf(theta_prop[3])*priors['q'].pdf(theta_prop[4])*priors['d'].pdf(theta_prop[5])/priors['phi'].pdf(theta[3])*priors['q'].pdf(theta[4])*priors['d'].pdf(theta[5])
    elif(m_prop==m and m==2):
        return 1.0

#-----------------------------------------------------------------------------
# This function generates the light curve for a signel lens.
#-----------------------------------------------------------------------------
def MT(t,u0,t0,te):
    mt = ((u0**2) + (((t-t0)/te)**2) + 2)/(np.sqrt((u0**2)+((t-t0)/te)**2)*np.sqrt(((u0**2) + ((t-t0)/te)**2)+4))
    return mt

#-----------------------------------------------------------------------------
# This function generates the light curve for a binary lens.
#-----------------------------------------------------------------------------
def binary2(t,u0,t0,te,phi,q,d):

    c = vbb.VBBinaryLensing()
    pr = vbb.doubleArray(6)
    pr[0] = np.log(d)
    pr[1] = np.log(q)
    pr[2] = u0
    pr[3] = phi
    pr[4] = np.log(1e-10)
    pr[5] = np.log(te)
    pr[6] = t0
    vbb.Tol = 1.e-3
    mag = np.empty_like(t)
    for i,ti in enumerate(t):
        mag[i] = c.BinaryLightCurve(pr,ti)
    return mag


def source(t,t0,te,u0,phi):

    phi = math.radians(phi)
    xs_binary = np.array([(i-t0)/te*np.cos(phi)+u0*np.sin(phi) for i in t])
    ys_binary = np.array([u0*np.cos(phi)-((i-t0)/te)*np.sin(phi) for i in t])

    return xs_binary,ys_binary

def binary(q,d,zs_real,zs_imag):

    # Initialisation of static variables:
    q = float(q)
    d = float(d)
    m2 = 1/(1+q)
    m1 = 1-m2
    z1 = d/abs(1+1/q)
    z2 = -z1/q
    zs = complex(zs_real,zs_imag)
    zs1 = zs.conjugate()-z1
    zs2 = zs.conjugate()-z2
    zsa = 2*zs.conjugate()-z1-z2
    zsm = zs1*zs2

    # Calculating Coefficients of C(Z) a 5th order polynomial.
    #-----------------------------------------------------------------------------
    a = -(z1+z2)
    b = z1*z2

    k0 = b**2
    k1 = 2*a*b
    k2 = ((a**2)+2*b)
    k3 = 2*a
    k4 = 1

    g0 = k0*zsm
    g1 = b*zsa + k1*zsm
    g2 = 1 + a*zsa + k2*zsm
    g3 = zsa + k3*zsm
    g4 = k4*zsm

    c1 = -g0*zs - k0*zs.conjugate()
    c2 = g0 -g1*zs -k1*zs.conjugate()-b
    c3 = g1 -g2*zs -k2*zs.conjugate()-a
    c4 = g2 -g3*zs -k3*zs.conjugate()-1
    c5 = g3 -k4*zs.conjugate()-g4*zs
    c6 = g4

    # Solving the 5th order polynomial.
    #-----------------------------------------------------------------------------
    c = [c6,c5,c4,c3,c2,c1]
    p = np.poly1d(c)
    z = roots(p)
    z_use = []

    # Check to see which roots satisfy the lens equation
    #-----------------------------------------------------------------------------
    for ii in z:
         check = zs - ii + (m1/(ii.conjugate()-z1.conjugate())) + (m2/(ii.conjugate()-z2.conjugate()))
         if(abs(check) <= abs(complex(1e-13,1e-13))):
             #This sets a threshold for the substitution check, to get around the computer error
             z_use.append(ii)

    # Using the solutions for image positions, calculate the magnitude.
    #-----------------------------------------------------------------------------
    A = 0;
    for jj in z_use:
        # First calculating the two components of the determinant
        delzs_delzbar = jj.conjugate()/((jj.conjugate() - z1.conjugate())*(jj.conjugate() - z2.conjugate())**2)- 1/((jj.conjugate()-z1.conjugate())*(jj.conjugate() - z2.conjugate()))+ jj.conjugate()/(((jj.conjugate()-z1.conjugate())**2)*(jj.conjugate()-z2.conjugate()))

        delzsbar_delz = jj/((jj-z1)*(jj-z2)**2) - 1/((jj-z1)*(jj-z2)) - jj.conjugate()/(((jj-z1)**2)*(jj-z2))
        # Calculating the determinant of J
        det_J = 1 - delzs_delzbar*delzsbar_delz
        #Calculating the Amplitude of the ith image
        A_i = 1/abs(det_J)
        A = A+A_i

    return A
