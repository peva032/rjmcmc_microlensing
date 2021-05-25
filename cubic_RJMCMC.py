import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import progressbar

dataQuad = pd.read_csv('quadratic_data.csv');
dataLin = pd.read_csv('linear_data.csv')
dataCub = pd.read_csv('cubic_data.csv')
dataChoice = input("Type 'l' for linear data, 'q' for Quadratic data or 'c' for Cubic data: ")
if dataChoice=='l':
    data = dataLin
    s = 'linear'
elif dataChoice=='q':
    data = dataQuad
    s = 'quadratic'
elif dataChoice=='c':
    data = dataCub
    s = 'cubic'

print('------------------------------------------------------------------')
print('      Running RJMCMC with noisy %s data'%s)
print('------------------------------------------------------------------')

x = data['x']
y = data['y']
#Models:
A1 = np.transpose([[x],[np.ones(len(x))]])
A2 = np.transpose([[x**2],[x],[np.ones(len(x))]])
A3 = np.transpose([[x**3],[x**2],[x],[np.ones(len(x))]])
A = {'1':A1,'2':A2,'3':A3}
sigmal = 1;
sigmae = 0.01;
M_store = []
M1_params = []
M2_params = []
M3_params = []
alpha = 0.1
count = 0
count1 = 0
count2 = 0
count3 = 0
#------------------------------------------------
#RJMCMC
#------------------------------------------------
def likelihood(M,theta):
    pot = np.matrix(y).reshape(len(y),1)-np.matrix(np.dot(A[str(M)],theta[-M-1:])).reshape(len(y),1)
    like = (1.0/(np.sqrt(2*np.pi*(sigmae**2))))*np.exp(-float(np.sqrt(np.dot(np.transpose(pot/sigmae),pot/sigmae)))/2.0)
    return like
def prior(M):
    return sigmal*np.random.randn(M+1)
def posterior_ratio(M_prop,M,theta,theta_prop):
    prior_ratio = np.prod(prior(M_prop))/np.prod(prior(M))
    likelihood_ratio = likelihood(M_prop,theta_prop)/likelihood(M,theta) # taking the maximum values of each likelihood before calculating the ratio
    if(M_prop == 2):
        #r_prime = 1.0
        r_prime = np.prod(prior(M_prop-1))
    elif(M_prop == 1):
        r_prime = r_prime = np.prod(prior(M_prop-1))
    elif(M_prop==3):
        r_prime = 1.0
    ratio_prob = likelihood_ratio*prior_ratio*(1/r_prime)
    return ratio_prob
#Random initialisation:
#theta = [[0],[0],[0]]
theta = [[0],[0],[0],[0]]
#M = np.random.randint(1,3)
M = np.random.randint(1,4)
N = 1000000
bar = progressbar.ProgressBar()

for i in bar(range(N)):
    #M_prop = np.random.randint(1,3)
    M_prop = np.random.randint(1,4)
    #theta_prop = theta+alpha*np.random.randn(3,1)
    theta_prop = theta+alpha*np.random.randn(4,1)
    P_accept = min(1,posterior_ratio(M_prop,M,theta,theta_prop))
    u = np.random.uniform(0,1)
    if(u <= P_accept):
        count = count+1
        M = M_prop
        theta=theta_prop
        M_store.append(M)

        if M==1:
            M1_params.append(theta[-M-1:])
            count1 = count1+1
        elif M==2:
            M2_params.append(theta[-M-1:])
            count2 = count2+1
        elif M==3:
            M3_params.append(theta[-M-1:])
            count3 = count3+1

M1_params = np.matrix(np.array(M1_params)).reshape(len(M1_params),2)
M2_params = np.matrix(np.array(M2_params)).reshape(len(M2_params),3)
M3_params = np.matrix(np.array(M3_params)).reshape(len(M3_params),4)
param1 = pd.DataFrame(M1_params)
param2 = pd.DataFrame(M2_params)
param3 = pd.DataFrame(M3_params)
theta_op1 = np.zeros(2)
theta_op2 = np.zeros(3)
theta_op3 = np.zeros(4)
#Calculating the conditional means for each parameter.
theta_op1[0] = np.mean(param1[0])
theta_op1[1] = np.mean(param1[1])

theta_op2[0] = np.mean(param2[0])
theta_op2[1] = np.mean(param2[1])
theta_op2[2] = np.mean(param2[2])

theta_op3[0] = np.mean(param3[0])
theta_op3[1] = np.mean(param3[1])
theta_op3[2] = np.mean(param3[2])
theta_op3[3] = np.mean(param3[3])

plt.figure(1)
plt.hist(M_store,bins=20)
plt.title('Histogram of models explored')
plt.xlabel('Model')
print("acceptance ratio: ",count/N)
print("Model 1 Probability: %f, with %d counts"%(count1/count,count1))
print("Model 2 Probability: %f, with %d counts"%(count2/count,count2))
print("Model 3 Probability: %f, with %d counts"%(count3/count,count3))
print('------------------------------------------------------------------')
#------------------------------------------------
x_data = np.linspace(-2,2,len(x))
plt.figure(2)
plt.plot(data['x'],data['y'],'ko',label='data')
plt.plot(x_data,theta_op3[0]*x_data**3+theta_op3[0]*x_data**2+theta_op3[1]*x_data+theta_op3[2],label='Cubic Model')
plt.plot(x_data,theta_op2[0]*x_data**2+theta_op2[1]*x_data+theta_op2[2],label='Quadratic Model')
plt.plot(x_data,theta_op1[0]*x_data+theta_op1[1],label='Linear Model')
plt.title('Noisy Data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
