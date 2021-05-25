import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from singlelc import MT
from binarylc import binary2
from scipy.stats import uniform
import math
import progressbar
from operator import add
import numpy.random as rnd

#------------------------------------------------------------------------------
#importing simulation data of Microlensing events.

sel = raw_input('To select single lens data press s, and to select binary lens data press b: ')
if sel=='s':
    data = pd.read_csv('single_data.csv')
    st = 'single'
elif sel=='b':
    data = pd.read_csv('binary_data.csv')
    st = 'binary'

#------------------------------------------------------------------------------
#Initialisation of model and parameters
t = data['t']
y = data['A']
N=1000
m_store = []
m = 2 #np.random.randint(1,3)
u0 = 0.2
t0 = 30
te = 10
phi = 20
q = 0.05
d = 1
theta = [[u0],[t0],[te],[phi],[q],[d]]
sigmae = 0.01 #Same as what was used for noise process in simulated data
#theta = [[rnd.uniform(0.001,2)],[rnd.uniform(5,60)],[rnd.uniform(1,20)]\
#,[rnd.uniform(0,360)],[rnd.uniform(10e-6,10e-1)],[rnd.uniform(0.01,3.0)]] #Initialisation of parameter vector
theta1_store = []
theta2_store = []
count1 = 0
count2 = 0
count = 0
m_store.append(m)

#------------------------------------------------------------------------------
#defining the likelihood based off additive gaussian noise
def likelihood(m,thet,t,y):
    if(m==1):
        pot = np.matrix(y).reshape(len(y),1)-np.matrix(MT(t,thet[0][0],thet[1][0],thet[2][0])).reshape(len(y),1)
    elif(m==2):
        pot = np.matrix(y).reshape(len(y),1)-np.matrix(binary2(t,thet[0][0],thet[1][0],thet[2][0],thet[3][0],thet[4][0],thet[5][0])).reshape(len(y),1)
    # cov = (sigmae**2)*np.identity(len(y))
    # like = (1.0/(np.sqrt(2*np.pi*(sigmae**2))))*np.exp(-float(np.sqrt(np.dot(np.dot(np.transpose(pot),cov),pot)))/2.0)
    like = (1.0/(np.sqrt(2*np.pi*(sigmae**2))))*np.exp(-float(np.sqrt(np.dot(np.transpose(pot/sigmae),pot/sigmae)))/2.0)
    return like

#Appears to be working fine.
#------------------------------------------------------------------------------
#defining prior distributions
def prior_u0(u0):
    A0 = [0.4023,0.591,0.0124]
    return (A0[0]*np.exp(-A0[1]*np.log(u0))+A0[2])
    #inverting since we want distribution for u0
def prior_t0(t0):
    DT = [5.558,1.8226]
    return np.exp(-((np.log(t0/0.0202)-DT[0])**2)/(2*DT[1]**2))
def prior_te(te):
    TE = [0.8284,6.0709,1.4283,0.1716,13.7152,1.3578]
    return TE[0]*np.exp(-((np.log(te/0.04)-TE[1])**2)/(2*TE[2]**2)) + TE[3]*np.exp(-((np.log(te/0.04)-TE[4])**2)/(2*TE[5]**2))
def prior_u0b(u0):
    p_u0b = uniform(0.001,4)
    return p_u0b.pdf(u0)
def prior_t0b(t0):
    p_t0b = uniform(1,100)
    return p_t0b.pdf(t0)
def prior_teb(te):
    p_teb = uniform(1,50)
    return p_teb.pdf(te)
def prior_phi(phi):
    p_phi = uniform(0,360)
    return p_phi.pdf(phi)
def prior_d(d):
    p_d = uniform(0.01,10.0)
    return p_d.pdf(d)
def prior_q(q):
    p_q = uniform(10e-6,1)
    return p_q.pdf(q)
#------------------------------------------------------------------------------
#defining posterior
def posterior_ratio(t,y,m,m_prop,theta,theta_prop):
    prior_ratio = 1
    r_u = prior_phi(rnd.uniform(0,180))*prior_d(rnd.uniform(0.01,2.0))*prior_q(rnd.uniform(10e-6,10e-1))
    #r_u is the distribution of auxiliary variables.
    if(m_prop>m):
        prior_ratio = float(prior_u0b(theta_prop[0][0])*prior_t0b(theta_prop[1][0])*prior_teb(theta_prop[2][0])*prior_phi(theta_prop[3][0])*prior_d(theta_prop[5][0])*prior_q(theta_prop[4][0]))/float(prior_u0b(theta[0][0])*prior_t0b(theta[1][0])*prior_teb(theta[2][0])*r_u)
    elif(m_prop<m):
        prior_ratio = float(prior_u0b(theta_prop[0][0])*prior_t0b(theta_prop[1][0])*prior_teb(theta_prop[2][0])*r_u)/float(prior_u0b(theta[0][0])*prior_t0b(theta[1][0])*prior_teb(theta[2][0])*prior_phi(theta[3][0])*prior_d(theta[5][0])*prior_q(theta[4][0]))
    elif(m_prop==1 & m==1):
        prior_ratio = float(prior_u0b(theta_prop[0][0])*prior_t0b(theta_prop[1][0])*prior_teb(theta_prop[2][0])*r_u)/float(prior_u0b(theta[0][0])*prior_t0b(theta[1][0])*prior_teb(theta[2][0])*r_u)
    elif(m_prop==2 & m==2):
        prior_ratio = float(prior_u0b(theta_prop[0][0])*prior_t0b(theta_prop[1][0])*prior_teb(theta_prop[2][0])*prior_phi(theta_prop[3][0])*prior_d(theta_prop[5][0])*prior_q(theta_prop[4][0]))/float(prior_u0b(theta[0][0])*prior_t0b(theta[1][0])*prior_teb(theta[2][0])*prior_phi(theta[3][0])*prior_d(theta[5][0])*prior_q(theta[4][0]))
#------------------------------------------------------------------------------
    post = (float(likelihood(m_prop,theta_prop,t,y))/float(likelihood(m,theta,t,y)))*float(prior_ratio)
    return post
#------------------------------------------------------------------------------
alpha = 0.9
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
theta_prop = []
for i in bar(range(N)):
    m_prop = 2
    #theta_prop = theta + alpha*rnd.randn(6,1)
    theta_prop = [[rnd.uniform(0.01,1)],[rnd.uniform(5,50)],[rnd.uniform(5,20)],[rnd.uniform(0,90)],[rnd.uniform(10e-6,10e-1)],[rnd.uniform(0.01,1.0)]]
    # condition1 = theta_prop[0][0]>0 and theta_prop[0][0] < 1
    # condition2 = theta_prop[1][0]>1 and theta_prop[1][0] < 50
    # condition3 = theta_prop[2][0]>1 and theta_prop[2][0] < 20
    # condition4 = theta_prop[3][0]>0 and theta_prop[3][0] < 180
    # condition5 = theta_prop[4][0]>10e-6 and theta_prop[4][0] < 10e-1
    # condition6 = theta_prop[5][0]>0.01 and theta_prop[5][0] < 3.0
    # condition = condition1 and condition2 and condition3 and condition4 and condition5 and condition6
    # if(condition):
    P_accept = min(1,posterior_ratio(t,y,m,m_prop,theta,theta_prop))
    u = np.random.uniform(0,1)

    if(u < P_accept):
        count = count+1
        m = m_prop
        theta=theta_prop
        m_store.append(m)

        if m==1:
            theta1_store.append(theta[:m*3])
            #print(theta[:m*3])
            count1 = count1+1
        elif m==2:
            theta2_store.append(theta[:m*3])
            #print(theta[:m*3])
            count2 = count2+1

# theta1_store = np.matrix(np.array(theta1_store)).reshape(len(theta1_store),3)
# theta2_store = np.matrix(np.array(theta2_store)).reshape(len(theta2_store),6)
theta1_store = np.array(theta1_store).reshape(len(theta1_store),3)
theta2_store = np.array(theta2_store).reshape(len(theta2_store),6)
param1 = pd.DataFrame(theta1_store)
param2 = pd.DataFrame(theta2_store)
theta_op1 = np.zeros(3)
theta_op2 = np.zeros(6)

#Calculating the conditional means for each parameter.
theta_op1[0] = np.mean(param1[0])
theta_op1[1] = np.mean(param1[1])
theta_op1[2] = np.mean(param1[2])

theta_op2[0] = np.mean(param2[0])
theta_op2[1] = np.mean(param2[1])
theta_op2[2] = np.mean(param2[2])
theta_op2[3] = np.mean(param2[3])
theta_op2[4] = np.mean(param2[4])
theta_op2[5] = np.mean(param2[5])

#------------------------------------------------------------------------------
#Plotting results:

print("acceptance ratio: ",count/float(N))
print("Model 1 Probability: %f, with %d counts"%(count1/float(count),count1))
print("Model 2 Probability: %f, with %d counts"%(count2/float(count),count2))
print('------------------------------------------------------------------')
#------------------------------------------------
if(1):
    plt.figure(1)
    plt.hist(m_store,bins=10)
    plt.title('Histogram of models explored')
    plt.xlabel('$Model$')

    t_data = np.linspace(0,60,len(t))
    plt.figure(2)
    plt.plot(data['t'],data['A'],'ko',label='data')
    plt.plot(t_data,MT(t_data,theta_op1[0],theta_op1[1],theta_op1[2]),label='single lens model')
    plt.plot(t_data,binary2(t_data,theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5]),label='binary lens model')
    plt.title('$Simulated\; data\; with\; rjmcmc\; model\; estimates$')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$A(t)$')
    plt.show()

    # plt.figure(3)
    # plt.hist(param2[0],bins=100)
    # plt.plot(np.ones(100)*theta_op2[0],np.linspace(0,6,100),'r-')
    # plt.plot(np.ones(100)*0.1,np.linspace(0,6,100),'k-')
    # plt.title('Histogram of u0 samples for m2')
    # plt.xlabel('$u_0$')
    # plt.show()
