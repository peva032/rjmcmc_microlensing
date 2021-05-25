import progressbar
from rjfun import *

#Selecting data for rjmcmc:
# sel = raw_input('To select single lens data press s, and to select binary lens data press b: ')
sel='b'
actualparams = [0.01,15,10,3.7168146928204138,0.001,0.6]

if sel=='s':
    data = pd.read_csv('single_data.csv')
    st = 'single'
    m = 1
elif sel=='b':
    data = pd.read_csv('binary_data.csv')
    st = 'binary'
    m = 2

# n=betaprior(1,3)
# k=np.linspace(0,1,1000);
# plt.plot(k,n.pdf(k))
# plt.hist([n.draw() for i in range(1000)],normed=True)
# n.dist.mean()
# plt.show()

# Initialisations:
t = np.array(data['t'])
y = np.array(data['A'])
N=10000
m_store = []
# m = np.random.randint(1,3)
#define priors
# u0 = uniprior(0,1)
# t0 = uniprior(0,40)
# te = uniprior(1e-4,30)
params = ['u0','t0','te','phi','q','d']#,'rate']
# rate = uniprior(2,10)
u0 = betaprior(1,5)
t0 = normprior(10,7)
te = normprior(12,5)
phi = uniprior(0,2*np.pi)
q = uniprior(0,1)
d = uniprior(0,1)
priors = {'u0':u0,'t0':t0,'te':te,'phi':phi,'q':q,'d':d}#,'rate':rate}

sigmae = 0.4
cov = sigmae**2 * np.eye(len(y))
theta = np.array([priors[p].draw() for p in params[0:m*3]])
# theta = actualparams
alpha = np.array([priors[p].range for p in params])/50.0
covp = np.diag(alpha**2)
theta1 = []
theta2 = []
m_store = []
count1 = 0
count2 = 0
count = 0


#==============================================================================
#RJMCMC algorithm

it = 0
bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
for i in bar(range(N)):
    it = it+1
    m_prop = rn.randint(1,3)
    theta_prop = proposal2(m,m_prop,theta,covp,priors,params)
    # theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
    # while u0.pdf(theta_prop[0])*t0.pdf(theta_prop[1])*te.pdf(theta_prop[2])*phi.pdf(theta_prop[3])*q.pdf(theta_prop[4])*d.pdf(theta_prop[5])==0.0:
    #     theta_prop = multivariate_normal.rvs(mean=theta, cov=covp)
    # theta_prop = [theta[i]+alpha[i]*np.random.randn() for i in range(len(alpha))]
    # condition =  u0.pdf(theta_prop[0])*t0.pdf(theta_prop[1])*te.pdf(theta_prop[2])*phi.pdf(theta_prop[3])*q.pdf(theta_prop[4])*d.pdf(theta_prop[5])>0.0
    # if(condition):
    #print(theta_prop[:m*3])
    if(not offsupport(m_prop,theta_prop,priors,params)):
        acc = likelihood_ratio(t,y,m,m_prop,theta,theta_prop,cov)*prior_ratio(m,m_prop,theta,theta_prop,priors) #likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov)
    else:
        acc = 0.0

    u = rn.uniform(0,1)
    if (u<=acc):
        count = count+1
        theta = theta_prop
        m = m_prop
        m_store.append(m)
        print(theta[:m*3])
        #print('Accepted')
        if m==1:
            theta1.append(theta[:m*3])
            count1=count1+1
        if m==2:
            theta2.append(theta[:m*3])
            count2 =count2+1
        print("Acc: %f, M1: %f, M2: %f"% ((count)/float(count1+count2),count1/float(count1+count2),count2/float(count1+count2)))
        # print("P: %f, LR: %f, close: %s" % (post_ratio(t,y,m,m_prop,theta,theta_prop,cov,priors)*prop_ratio(m,m_prop,theta,theta_prop,priors), likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov), str(np.isclose(post_ratio(t,y,m,m_prop,theta,theta_prop,cov,priors)*prop_ratio(m,m_prop,theta,theta_prop,priors),likelihood(m_prop,t,y,theta_prop,cov)/likelihood(m,t,y,theta,cov)))))
    else:
        m_store.append(m)
        # print(theta[:m*3])
        #print('Accepted')
        if m==1:
            theta1.append(theta[:m*3])
            count1=count1+1
        if m==2:
            theta2.append(theta[:m*3])
            count2 =count2+1
        # print("Acc: %f, M1: %f, M2: %f, u: %f"% ((count)/float(it+1),count1/float(count1+count2),count2/float(count1+count2),acc))

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
print("Model 1 Probability: %f, with %d counts"%(count1/float(count1+count2),count1))
if(count==0.0 or count1==0.0):
    print('probability is zero for model 1')
if(count==0.0 or count2==0.0):
    print('probability is zero for model 2')
print("Model 2 Probability: %f, with %d counts"%(count2/float(count1+count2),count2))
print('------------------------------------------------------------------')
print(theta_op1[0],theta_op1[1],theta_op1[2])
print(theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5])
print('------------------------------------------------------------------')
if(1):
    plt.figure(1)
    plt.hist(m_store,bins=10)
    plt.title('Histogram of models explored')
    plt.xlabel('$Model$')
    t_data = np.linspace(min(t),max(t),2*len(t))
    plt.figure(2)
    plt.plot(data['t'],data['A'],'ko',label='data',markersize=0.8)
    plt.plot(t_data,MT(t_data,theta_op1[0],theta_op1[1],theta_op1[2]),'r--',label='single lens model',linewidth=0.5)
    plt.plot(t_data,binary2(t_data,theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5]),'b--',label='binary lens model',linewidth=0.5)
    plt.axis([t_data[0],t_data[-1],0,1.2*max(max(data['A']),MT(theta_op1[1],theta_op1[0],theta_op1[1],theta_op1[2]),binary2([theta_op2[1]],theta_op2[0],theta_op2[1],theta_op2[2],theta_op2[3],theta_op2[4],theta_op2[5]))])
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
        plt.vlines(actualparams[0],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(u0.left,u0.right)
        plt.subplot(312)
        plt.xlabel('t0')
        hist, bins = np.histogram(param1['t0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[1],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(t0.left,t0.right)
        plt.subplot(313)
        plt.xlabel('te')
        hist, bins = np.histogram(param1['te'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[2],[0],[100],'r')
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
        plt.vlines(actualparams[0],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(u0.left,u0.right)
        plt.subplot(322)
        plt.xlabel('t0')
        hist, bins = np.histogram(param2['t0'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[1],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(t0.left,t0.right)
        plt.subplot(323)
        plt.xlabel('te')
        hist, bins = np.histogram(param2['te'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[2],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(te.left,te.right)
        plt.subplot(324)
        plt.xlabel('phi')
        hist, bins = np.histogram(param2['phi'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[3],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(phi.left,phi.right)
        plt.subplot(325)
        plt.xlabel('q')
        hist, bins = np.histogram(param2['q'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[4],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(q.left,q.right)
        plt.subplot(326)
        plt.xlabel('d')
        hist, bins = np.histogram(param2['d'], bins=15, density=True)
        widths = np.diff(bins)
        plt.bar(bins[:-1], hist, widths)
        plt.vlines(actualparams[5],[0],[100],'r')
        plt.ylim((0,1.2*max(hist)))
        plt.xlim(d.left,d.right)
        plt.show()

filekey = {'b':'binary','s':'single'}
param1.to_csv(filekey[sel]+"_rjmcmc_output_1.csv")
param2.to_csv(filekey[sel]+"_rjmcmc_output_2.csv")
