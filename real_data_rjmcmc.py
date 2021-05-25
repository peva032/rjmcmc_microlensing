import progressbar
import time
from rjfun import *
from binary_data import generatebinary
from operator import add

# Initialisations:
data = pd.read_csv('OB110251.csv',sep='\t')
data.columns = ['t','A','sigmaA']
t = data['t']
A = data['A']
sigmaA = data['sigmaA']
t  = t - t[0]
t = np.array(t[1400:3450])
y = np.array(A[1400:3450])
sigmaA = sigmaA[1400:3450]
st = 'binary'
m = 2


N=5000
m_store = []
m = np.random.randint(1,3)

params = ['u0','t0','te','phi','q','d']
u0 = betaprior(1,3)
t0 = normprior(500,50)
te = normprior(80,20)
phi = uniprior(0,2*np.pi)
q = uniprior(0,0.1)
d = uniprior(0.5,1.5)
priors = {'u0':u0,'t0':t0,'te':te,'phi':phi,'q':q,'d':d}#,'rate':rate}

# sigmae = 4
cov = np.diag(sigmaA**2)
# theta = [priors[p].draw() for p in params]#
theta = [5.57e-2,500,63.439,4.4771,1.594e-3,1.346]
# theta = actualparams
alpha = np.array([priors[p].range for p in params])/50.0
theta1 = []
theta2 = []
theta1all=[]
thetaall=[]
m_store = []
count1 = 0
count2 = 0
count = 0
covp1 = np.diag((alpha/np.array([1.0,1.0,1.0,2.0,2.0,2.0]))**2)
covp2 = np.diag(alpha**2)

#==============================================================================
#RJMCMC algorithm
it = 0
maxlr=0
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
for i in bar(range(N)):
    it = it+1
    covp = np.diag(alpha**2)#(alpha**2)*np.identity(len(theta))
    # # theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
    # # while u0.pdf(theta_prop[0])*t0.pdf(theta_prop[1])*te.pdf(theta_prop[2])*phi.pdf(theta_prop[3])*q.pdf(theta_prop[4])*d.pdf(theta_prop[5])==0.0:
    # #     theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
    # # theta_prop = [theta[i]+alpha[i]*np.random.randn() for i in range(len(alpha))]
    # m_prop = rn.randint(1,3)
    # theta_prop = proposal(theta,covp,priors,params)
    # if((i%3==0) and (i>500)):
    #     if (np.shape(theta1all)[0]>100):
    #         covp1 = (2.83)**2*np.cov(np.transpose(theta1all))/6 + (0.1)**2*np.eye(6)/6
    #     if (np.shape(theta2)[0]>100):
    #         covp2 = (2.83)**2*np.cov(np.transpose(theta2))/6 + (0.1)**2*np.eye(6)/6
    m_prop = rn.randint(1,3)
    # if (m_prop==1):
    #     covp=covp1
    # else:
    #     covp=covp2
    theta_prop = proposal(theta,covp,priors,params)
    if(not offsupport(m_prop,theta_prop,priors,params)):
        # print('on support')
        while(True):
            try:
                # print('calculate likelihood_ratio')
                acc = likelihood_ratio(t,y,m,m_prop,theta,theta_prop,cov)*prior_ratio(m,m_prop,theta,theta_prop,priors)
                break
            except TimeoutError:
                print('timed out, drawing new proposal')
                theta_prop = proposal(theta,covp,priors,params)

    else:
        # print('off support')
        acc = 0.0
    u = rn.uniform(0,1)
    if u<=acc:
        count = count+1
        theta = theta_prop
        m = m_prop
        m_store.append(m)
        print(theta[:m*3])
        #print('Accepted')
        if m==1:
            theta1.append(theta[:m*3])
            theta1all.append(theta)
            count1=count1+1
        if m==2:
            theta2.append(theta[:m*3])
            count2 =count2+1
        thetaall.append(theta)
        print("Acc Rate: %f, m_prop: %d, acc %f, M1: %f, M2: %f"% ((count)/float(count1+count2),m_prop,acc,count1/float(count1+count2),count2/float(count1+count2)))
        # print("P: %f, LR: %f, close: %s" % (post_ratio(t,y,m,m_prop,theta,theta_prop,cov,priors)*prop_ratio(m,m_prop,theta,theta_prop,priors), likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov), str(np.isclose(post_ratio(t,y,m,m_prop,theta,theta_prop,cov,priors)*prop_ratio(m,m_prop,theta,theta_prop,priors),likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov)))))
    else:
        m_store.append(m)
        # print(theta[:m*3])
        #print('Accepted')
        if m==1:
            theta1.append(theta[:m*3])
            theta1all.append(theta)
            count1=count1+1
        if m==2:
            theta2.append(theta[:m*3])
            count2 =count2+1
        thetaall.append(theta)
        print("Acc Rate: %f, m_prop: %d, acc %f, M1: %f, M2: %f"% ((count)/float(count1+count2),m_prop,acc,count1/float(count1+count2),count2/float(count1+count2)))

theta1 = np.array(theta1).reshape(len(theta1),3)
theta2 = np.array(theta2).reshape(len(theta2),6)
param1 = pd.DataFrame(theta1,columns = ["u0", "t0", "te"])
param2 = pd.DataFrame(theta2,columns = ["u0", "t0", "te","phi","q","d"])
theta_op1 = np.zeros(3)
theta_op2 = np.zeros(6)

theta_op1[0] = np.mean(param1['u0'])
theta_op1[1] = np.mean(param1['t0'])
theta_op1[2] = np.mean(param1['te'])

theta_op2[0] = np.mean(param2['u0'])
theta_op2[1] = np.mean(param2['t0'])
theta_op2[2] = np.mean(param2['te'])
theta_op2[3] = np.mean(param2['phi'])
theta_op2[4] = np.mean(param2['q'])
theta_op2[5] = np.mean(param2['d'])

print("acceptance ratio: ",count/float(N))
print("Model 1 Probability: %f, with %d counts"%(count1/float(count),count1))
if(count==0.0 or count1==0.0):
    print('probability is zero for model 1')
if(count==0.0 or count2==0.0):
    print('probability is zero for model 2')
print("Model 2 Probability: %f, with %d counts"%(count2/float(count),count2))
print('------------------------------------------------------------------')
print(theta_op1[0],theta_op1[1],theta_op1[2])
print(theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5])
print('------------------------------------------------------------------')
if(0):
    plt.figure(1)
    plt.hist(m_store,bins=10)
    plt.title('Histogram of models explored')
    plt.xlabel('$Model$')
    t_data = np.linspace(0,800,len(t))
    plt.figure(2)
    plt.plot(data['t'],data['A'],'ko',label='data',markersize=0.8)
    plt.plot(t_data,MT(t_data,theta_op1[0],theta_op1[1],theta_op1[2]),'r--',label='single lens model',linewidth=0.5)
    plt.plot(t_data,binary2(t_data,theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5]),'b--',label='binary lens model',linewidth=0.5)
    # plt.axis([-10,100,0,13])
    plt.title('$Simulated\; data\; with\; rjmcmc\; model\; estimates$')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$A(t)$')
    if(sel=='s'):
        plt.figure(3)
        plt.subplot(311)
        plt.xlabel('u0')
        hist, bins = np.histogram(param1['u0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[0],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(u0.left,u0.right)
        plt.subplot(312)
        plt.xlabel('t0')
        hist, bins = np.histogram(param1['t0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[1],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(t0.left,t0.right)
        plt.subplot(313)
        plt.xlabel('te')
        hist, bins = np.histogram(param1['te'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[2],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(te.left,te.right)
        plt.show()
    if(sel=='b'):
        plt.figure(3)
        plt.subplot(321)
        plt.xlabel('u0')
        hist, bins = np.histogram(param2['u0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[0],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(u0.left,u0.right)
        plt.subplot(322)
        plt.xlabel('t0')
        hist, bins = np.histogram(param2['t0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[1],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(t0.left,t0.right)
        plt.subplot(323)
        plt.xlabel('te')
        hist, bins = np.histogram(param2['te'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[2],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(te.left,te.right)
        plt.subplot(324)
        plt.xlabel('phi')
        hist, bins = np.histogram(param2['phi'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[3],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(phi.left,phi.right)
        plt.subplot(325)
        plt.xlabel('q')
        hist, bins = np.histogram(param2['q'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[4],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(q.left,q.right)
        plt.subplot(326)
        plt.xlabel('d')
        hist, bins = np.histogram(param2['d'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        # plt.vlines(actualparams[5],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(d.left,d.right)
        plt.show()

filekey = {'b':'binary','s':'single'}
param1.to_csv("outputs/realdata/output_(single).csv")
param2.to_csv("outputs/realdata/output_(binary).csv")
mdf = pd.DataFrame(m_store,columns=["m"])
mdf.to_csv("outputs/realdata/mtrace.csv")
