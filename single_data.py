import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sigmal = 1
sigmae = 0.3
N=650
np.random.seed(1)
#generate single lens data
u0 = 0.01; te = 10;t0 = 15;
t = np.random.uniform(0,60,N)
tt = np.linspace(0,60,1000)
e = np.random.normal(0,sigmae,N)

def MT(t,u0,t0,te):

    mt = ((u0**2) + (((t-t0)/te)**2) + 2)/(np.sqrt((u0**2)+((t-t0)/te)**2)*np.sqrt(((u0**2) + ((t-t0)/te)**2)+4))

    return mt


y = MT(t,u0,t0,te) + e;
data = {'t':t,'A':y}
single_data = pd.DataFrame(data)
single_data.to_csv('single_data.csv',index=False)

if(1):
    plt.figure()
    plt.plot(t,y,"k.",markersize=0.7)
    plt.plot(tt,MT(tt,u0,t0,te),'b--',linewidth=0.5)
    plt.show()
