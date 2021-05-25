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


sel = raw_input('To select single lens data press s, and to select binary lens data press b: ')
if sel=='s':
    data = pd.read_csv('single_data.csv')
    st = 'single'
    m = 1

elif sel=='b':
    data = pd.read_csv('binary_data.csv')
    st = 'binary'
    m = 2

t = data['t']
y = data['A']
N=1000000
Nburn=int(N*0.7)
u0 = 3
t0 = 30
te = 20
phi = 20
q = 0.05
d = 1
theta = [u0,t0,te,phi,q,d]
theta_prop = []
theta_store = []
post_store = []
count = 0
sigmae = 0.01
#==============================================================================
#Tuning parameter
if m==1:
    alpha = 0.5
if m==2:
    alpha = 0.5
#==============================================================================

def likelihood(m,theta,t,y):
    if(m==1):
        pot = y - MT(t,theta[0],theta[1],theta[2])
    elif(m==2):
        pot = y - binary2(t,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5])
    cov = (sigmae**2)*np.identity(len(y))
    like = (1.0/(np.sqrt(2*np.pi*(sigmae**2))))*np.exp(-float(np.sqrt(np.dot(np.dot(np.transpose(pot),cov),pot)))/2.0)
    return like

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
    p_u0b = uniform(0,10)
    return p_u0b.pdf(u0)
def prior_t0b(t0):
    p_t0b = uniform(0,60)
    return p_t0b.pdf(t0)
def prior_teb(te):
    p_teb = uniform(0,30)
    return p_teb.pdf(te)
def prior_phi(phi):
    p_phi = uniform(0,360)
    return p_phi.pdf(phi)
def prior_d(d):
    p_d = uniform(0.01,5.0)
    return p_d.pdf(d)
def prior_q(q):
    p_q = uniform(10e-6,10)
    return p_q.pdf(q)
def prop(m,theta):
    if m==1:
        cov = (alpha**2)*np.identity(len(theta))
        var = multivariate_normal(theta,cov)
        return var.pdf(theta)
    if m==2:
        return np.prod(np.random.normal(theta[0],alpha)*np.random.normal(theta[1],alpha)*np.random.normal(theta[2],alpha)\
        *np.random.normal(theta[3],alpha)*np.random.normal(theta[4],alpha)*np.random.normal(theta[5],alpha))


def posterior_ratio(m,theta,theta_prop,t,y):

    like_ratio = float(likelihood(m,theta_prop,t,y))/float(likelihood(m,theta,t,y))
    if m==1:
        prior_ratio = float(prior_u0(theta_prop[0])*prior_t0(theta_prop[1])*prior_te(theta_prop[2]))\
        /float(prior_u0(theta[0])*prior_t0(theta[1])*prior_te(theta[2]))
    elif m==2:
        prior_ratio = float(prior_u0b(theta_prop[0])*prior_t0b(theta_prop[0])\
        *prior_teb(theta_prop[2])*prior_phi(theta_prop[3])*prior_q(theta_prop[4])*prior_d(theta_prop[5]))\
        /float(prior_u0b(theta[0])*prior_t0b(theta[1])\
        *prior_teb(theta[2])*prior_phi(theta[3])*prior_q(theta[4])*prior_d(theta[5]))
    post = like_ratio*prior_ratio
    return post
itr = 0
#===============================================================================
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
for i in bar(range(N)):
    itr = itr+1

    theta_prop =theta + np.random.normal(0,alpha,size=6)
    # theta_prop[1]=10.0
    # theta_prop[2]=10.0
    if(np.all(theta_prop>0)):
        #print(theta_prop)
        acc = min(1,posterior_ratio(m,theta,theta_prop,t,y)*(prop(m,theta)/prop(m,theta_prop)))
        u = rn.uniform(0,1)
        if acc>=u:
            theta = theta_prop
            #print(acc)
            count = count+1
            post_store.append(likelihood(m,theta,t,y)*float(prior_u0(1.0/theta[0])*prior_t0(theta[1])*prior_te(theta[2])))
            #if itr>=Nburn:
            theta_store.append(theta[:m*3])
#==============================================================================
if(m==1):
    xinit = [0.2,15,5]
    xinit = np.array(xinit)
    objective = lambda x: (y - MT(t,x[0],x[1],x[2]))
    plsq = leastsq(objective, xinit)
    print(plsq)
if(m==2):
    xinit = [u0,t0,te,phi,q,d]
    xinit = np.array(xinit)
    objective = lambda x: (y - binary2(t,x[0],x[1],x[2],x[3],x[4],x[5]))
    plsq = leastsq(objective, xinit)
    print(plsq)
#==============================================================================

if m==1:
    theta_store = np.array(theta_store).reshape(len(theta_store),3)
    param = pd.DataFrame(theta_store)
    theta_op = np.zeros(3)
    # theta_op[0] = param[0][np.argmax(post_store)]
    # theta_op[1] = param[1][np.argmax(post_store)]
    # theta_op[2] = param[2][np.argmax(post_store)]
    theta_op[0] = np.mean(param[0])
    theta_op[1] = np.mean(param[1])
    theta_op[2] = np.mean(param[2])
    print(theta_op[0],theta_op[1],theta_op[2])
if m==2:
    theta_store = np.array(theta_store).reshape(len(theta_store),6)
    param = pd.DataFrame(theta_store)
    theta_op = np.zeros(6)
    theta_op[0] = np.mean(param[0])
    theta_op[1] = np.mean(param[1])
    theta_op[2] = np.mean(param[2])
    theta_op[3] = np.mean(param[3])
    theta_op[4] = np.mean(param[4])
    theta_op[5] = np.mean(param[5])
    print(theta_op[0],theta_op[1],theta_op[2],theta_op[3],theta_op[4],theta_op[5])

print("acceptance ratio: ",count/float(N))
print('------------------------------------------------------------------')
if m==1:
    t_data = np.linspace(0,60,len(t))
    plt.figure(1)
    plt.plot(data['t'],data['A'],'ko',label='data')
    plt.plot(t_data,MT(t_data,theta_op[0],theta_op[1],theta_op[2]),label='single lens model')
    plt.plot(t_data,MT(t_data,plsq[0][0],plsq[0][1],plsq[0][2]),label='leastsq')
    plt.title('$Simulated\; data\; with\; MCMC\; estimates$')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$A(t)$')
    plt.show()

    plt.figure(2)
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[0],label='$u_0$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[1],label='$t_0$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[2],label='$t_e$')
    plt.legend()
    plt.xlabel('Iteration')
    plt.show()

if m==2:
    t_data = np.linspace(0,60,len(t))
    plt.figure(1)
    plt.plot(data['t'],data['A'],'ko',label='data')
    plt.plot(t_data,binary2(t_data,theta_op[0],theta_op[1],theta_op[2],theta_op[3],theta_op[4],theta_op[5]),label='binary lens model')
    plt.plot(t_data,binary2(t_data,plsq[0][0],plsq[0][1],plsq[0][2],plsq[0][3],plsq[0][4],plsq[0][5]),label='leastsq')
    plt.title('$Simulated\; data\; with\; MCMC\; estimates$')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$A(t)$')
    plt.show()

    plt.figure(2)
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[0],label='$u_0$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[1],label='$t_0$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[2],label='$t_e$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[3],label='$phi$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[4],label='$q$')
    plt.plot(np.linspace(1,len(param[0]),len(param[0])),param[5],label='$d$')
    plt.legend()
    plt.xlabel('Iteration')
    plt.show()
