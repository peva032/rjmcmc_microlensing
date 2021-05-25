import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from quadratic_data import quad_data
from linear_data import lin_data
# For loop put in to run test simulation for large number of times.
simN = 1
count1_store = []
count2_store = []
count_store = []
param_store1 = []
param_store2 = []
model_store = []
mprob_store = []

bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])

for i in bar(range(simN)):

    dataChoice = 1#np.random.randint(1,3)
    if dataChoice == 1:
        param_store1.append(lin_data())
        print(param_store1)
        dataLin = pd.read_csv('linear_data.csv')
        data = dataLin
        s = 'linear'
    elif dataChoice == 2:
        param_store2.append(quad_data())
        print(param_store2)
        dataQuad = pd.read_csv('quadratic_data.csv');
        data = dataQuad
        s = 'quadratic'

    # print('------------------------------------------------------------------')
    # print('      Running RJMCMC with noisy %s data'%s)
    # print('------------------------------------------------------------------')

    x = data['x']
    y = data['y']
    #Models:
    A1 = np.transpose([[x],[np.ones(len(x))]])
    A2 = np.transpose([[x**2],[x],[np.ones(len(x))]])
    A = {'1':A1,'2':A2}
    sigmal = 1;
    sigmae = 0.2;
    M_store = []
    M1_params = []

    M2_params = []
    count = 0
    count1 = 0
    count2 = 0
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
        likelihood_ratio = likelihood(M_prop,theta_prop)/likelihood(M,theta)
        r1 = np.prod(prior(M_prop-1)) #Auxilary variable distribution for model 1
        r2 = 1.0 #Auxilary variable distribution for model 2 which has Auxilary variable that is the empty set
        #Thus we use the identity as our mapping function
        if(M_prop == 2):
            ratio_prob = likelihood_ratio*prior_ratio*(r1/r2)
        elif(M_prop == 1):
            ratio_prob = likelihood_ratio*prior_ratio*(r2/r1)
        return ratio_prob
    #Initialisation:
    theta = [[np.random.uniform(0,1)],[np.random.uniform(0,1)],[np.random.uniform(0,1)]]
    M = 1
    alpha = 0.1
    N = 10000

    for i in np.arange(1,N):
        M_prop = np.random.randint(1,3)
        theta_prop = theta+alpha*np.random.randn(3,1)
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

    M1_params = np.matrix(np.array(M1_params)).reshape(len(M1_params),2)
    M2_params = np.matrix(np.array(M2_params)).reshape(len(M2_params),3)
    param1 = pd.DataFrame(M1_params)
    param2 = pd.DataFrame(M2_params)
    theta_op1 = np.zeros(2)
    theta_op2 = np.zeros(3)
    #Calculating the conditional means for each parameter.
    theta_op1[0] = np.mean(param1[0])
    theta_op1[1] = np.mean(param1[1])

    theta_op2[0] = np.mean(param2[0])
    theta_op2[1] = np.mean(param2[1])
    theta_op2[2] = np.mean(param2[2])

    count1_store.append(count1)
    count2_store.append(count2)
    count_store.append(count)
    model_store.append(dataChoice)
    if dataChoice==1:
        mprob_store.append(float(count1)/float(count))
    elif dataChoice==2:
        mprob_store.append(float(count2)/float(count))

#------------------------------------------------------------------------------

    # print("acceptance ratio: ",count/float(N))
    # print("Model 1 Probability: %f, with %d counts"%(count1/float(count),count1))
    # print("Model 2 Probability: %f, with %d counts"%(count2/float(count),count2))
    # print('------------------------------------------------------------------')
    #------------------------------------------------
    if(1):
        plt.figure(1)
        plt.hist(M_store,bins=15,color='gray')
        plt.title('$Histogram\; of\; models\; explored$')
        plt.xlabel('$Model$')

        x_data = np.linspace(-2,2,len(x))
        plt.figure(2)
        plt.plot(data['x'],data['y'],'ko',label='data',markersize=0.7)
        plt.plot(x_data,theta_op2[0]*x_data**2+theta_op2[1]*x_data+theta_op2[2],label='Quadratic estimate',linewidth=0.5)
        plt.plot(x_data,theta_op1[0]*x_data+theta_op1[1],label='Linear estimate',linewidth=0.5)
        plt.title('$Simulated\; Data$')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        plt.figure(3)
        plt.plot(np.linspace(1,count,count),M_store,'ko',fillstyle='none')
        plt.plot(np.linspace(1,count,count),M_store,'b-')
        plt.title('$Markov\; chain\; model\; samples$')
        plt.xlabel('$Iteration$')
        plt.ylabel('$Model\; index$')
        plt.axis([0,count,0.5,2.5])
        plt.show()

if(0):
    plt.figure(5)
    plt.hist(model_choice,bins=20)
    # plt.title("$Histogram\; of\; models\; explored\; over\;$"+str(simN)+"$\; simulations,\; with$"+s+"$\; data$")
    plt.xlabel('Model')
    plt.ylabel("Frequency")
    plt.show()

if(1):

    plt.figure(6)
    plt.plot(np.linspace(1,len(param1[0]),len(param1[0])),np.ones(len(param1[0]))*1.6243453636632417,'r--',label='true value')
    plt.plot(np.linspace(1,len(param1[0]),len(param1[0])),param1[0],label='parameter-1')
    plt.plot(np.linspace(1,len(param1[0]),len(param1[0])),param1[1],'k',label='parameter-2')
    plt.plot(np.linspace(1,len(param1[0]),len(param1[0])),np.ones(len(param1[0]))*-0.6117564136500754,'r--')
    plt.legend()
    plt.xlabel('$Iteration$')
    plt.title('$Linear\; model\; samples$')
    plt.show()

    plt.figure(7)
    plt.plot(np.linspace(1,len(param2[0]),len(param2[0])),param2[0],label='parameter-1')
    plt.plot(np.linspace(1,len(param2[0]),len(param2[0])),np.ones(len(param2[0]))*-0.5497461766598448,'r--',label='true value')
    plt.plot(np.linspace(1,len(param2[0]),len(param2[1])),param2[1],'k',label='parameter-2')
    plt.plot(np.linspace(1,len(param2[0]),len(param2[0])),np.ones(len(param2[0]))*-1.4028727151691185,'r--')
    plt.plot(np.linspace(1,len(param2[0]),len(param2[0])),param2[2],'gray',label='parameter-3')
    plt.plot(np.linspace(1,len(param2[0]),len(param2[0])),np.ones(len(param2[0]))*1.582752300825677,'r--')
    plt.legend()
    plt.xlabel('$Iteration$')
    plt.title('$Quadratic\; model\; samples$')
    plt.show()

# d = {'model data':model_store,'model_probability':mprob_store,'count1':count1_store,'count2':count2_store,'count':count_store}
# df = pd.DataFrame(d)
# print(df.head())
# #df.to_csv('ql_model_data.csv',index=False)
