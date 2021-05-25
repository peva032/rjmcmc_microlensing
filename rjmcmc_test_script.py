# for testing only!!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binarylc import binary2
from single_data import MT
import numpy.random as rnd
from scipy.stats import uniform

m = 1
m_prop = 1
sigmae = 0.01
theta_prop = [[1.1177398445083613], [91.64502862686118], [32.43674423837054], [140.40277709084864], [0.4859958071902388], [3.6298197926906397]]
theta = [[0.1], [15], [10], [10], [0.09], [1]]

data = pd.read_csv('single_data.csv')

t = data['t']
y = data['A']

def likelihood(m,theta,t,y):
    if(m==1):
        pot = y - MT(t,theta[0][0],theta[1][0],theta[2][0])
    elif(m==2):
        pot = y - binary2(t,theta[0][0],theta[1][0],theta[2][0],theta[3][0],theta[4][0],theta[5][0])
    cov = sigmae*np.identity(len(y))
    like = (1.0/(np.sqrt(2*np.pi*(sigmae**2))))*np.exp(-float(np.sqrt(np.dot(np.dot(np.transpose(pot),cov),pot)))/2.0)
    return like

print(likelihood(m_prop,theta_prop,t,y))

def prior_u0(a0):
    A0 = [0.4023,0.591,0.0124]
    return (A0[0]*np.exp(-A0[1]*np.log(a0))+A0[2])
    #inverting since we want distribution for u0
def prior_t0(dt):
    DT = [5.558,1.8226]
    return np.exp(-((np.log(dt/0.0202)-DT[0])**2)/(2*DT[1]**2))
def prior_te(te):
    TE = [0.8284,6.0709,1.4283,0.1716,13.7152,1.3578]
    return TE[0]*np.exp(-((np.log(te/0.04)-TE[1])**2)/(2*TE[2]**2)) + TE[3]*np.exp(-((np.log(te/0.04)-TE[4])**2)/(2*TE[5]**2))
def prior_aux(m):
    return np.random.randn(m)
    #returning m uniform Probabilities for Auxilary variables
def prior_phi(phi):
    p_phi = uniform(0,360)
    return p_phi.pdf(phi)
def prior_d(d):
    p_d = uniform(0.01,6.0)
    return p_d.pdf(d)
def prior_q(q):
    p_q = uniform(10e-6,10e-1)
    return p_q.pdf(q)
#------------------------------------------------------------------------------
#defining posterior
def posterior_ratio(t,y,m,m_prop,theta,theta_prop):
    prior_ratio = 1
    r_u = prior_phi(rnd.uniform(0,360))*prior_d(rnd.uniform(0.01,6.0))*prior_q(rnd.uniform(10e-4,10e-2))
    #r_u is the distribution of auxiliary variables.
    if(m_prop>m):
        prior_ratio = float(prior_u0(1.0/theta_prop[0][0])*prior_t0(theta_prop[1][0])*prior_te(theta_prop[2][0])*prior_phi(theta_prop[3][0])*prior_d(theta_prop[5][0])*prior_q(theta_prop[4][0]))/float(prior_u0(1.0/theta[0][0])*prior_t0(theta[1][0])*prior_te(theta[2][0])*r_u)
        print(prior_u0(1/theta[0][0])*prior_t0(theta[1][0])*prior_te(theta[2][0])*r_u)
    elif(m_prop<m):
        prior_ratio = float(prior_u0(1.0/theta_prop[0][0])*prior_t0(theta_prop[1][0])*prior_te(theta_prop[2][0])*r_u)/float(prior_u0(1.0/theta[0][0])*prior_t0(theta[1][0])*prior_te(theta[2][0])*prior_phi(theta[3][0])*prior_d(theta[5][0])*prior_q(theta[4][0]))
    elif(m_prop==1 & m==1):
        prior_ratio = float(prior_u0(1.0/theta_prop[0][0])*prior_t0(theta_prop[1][0])*prior_te(theta_prop[2][0])*r_u)/float(prior_u0(1.0/theta[0][0])*prior_t0(theta[1][0])*prior_te(theta[2][0])*r_u)
    elif(m_prop==2 & m==2):
        prior_ratio = float(prior_u0(1.0/theta_prop[0][0])*prior_t0(theta_prop[1][0])*prior_te(theta_prop[2][0])*prior_phi(theta_prop[3][0])*prior_d(theta_prop[5][0])*prior_q(theta_prop[4][0]))/float(prior_u0(1.0/theta[0][0])*prior_t0(theta[1][0])*prior_te(theta[2][0])*prior_phi(theta[3][0])*prior_d(theta[5][0])*prior_q(theta[4][0]))
#--------------------------------------------------------------------------
    print("prior ratio: %f"%prior_ratio)
    post = (likelihood(m_prop,theta_prop,t,y)/likelihood(m,theta,t,y))*prior_ratio
    return post
#print(prior_u0(1.0/float(theta_prop[0][0])))
# print(prior_t0(theta_prop[1][0]))
# print(prior_te(theta_prop[2][0]))
# print(prior_phi(theta_prop[3][0]))
# print(prior_q(theta_prop[4][0]))
# print(prior_d(theta_prop[5][0]))

print("likelihood of m: %f"%likelihood(m,theta,t,y))
print("likelihood of m_prop: %f"%likelihood(m_prop,theta_prop,t,y))
print("Posterior ratio: %f"%posterior_ratio(t,y,m,m_prop,theta,theta_prop))
