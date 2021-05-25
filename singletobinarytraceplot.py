import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from rjfun import *

pwd = os.getcwd()
su0 = []
st0 = []
ste = []
bu0 = []
bt0 = []
bte =[]
bphi = []
bq = []
bd = []
actualparams = [0.1,15,10,1.047,0.01,0.5]

sin_data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_1_output_(single).csv')
bin_data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_1_output_(binary).csv')
sin_data2 = pd.read_csv(pwd+'/outputs/singletobinary/fig2_singletobinary_1_output_(single).csv')
bin_data2 = pd.read_csv(pwd+'/outputs/singletobinary/fig2_singletobinary_1_output_(binary).csv')

su0.append(sin_data['u0'])
su0.append(sin_data2['u0'])
st0.append(sin_data['t0'])
st0.append(sin_data2['t0'])
ste.append(sin_data['te'])
ste.append(sin_data2['te'])
bu0.append(bin_data['u0'])
bu0.append(bin_data2['u0'])
bt0.append(bin_data['t0'])
bt0.append(bin_data2['t0'])
bte.append(bin_data['te'])
bte.append(bin_data2['te'])
bphi.append(bin_data['phi'])
bphi.append(bin_data2['phi'])
bq.append(bin_data['q'])
bq.append(bin_data2['q'])
bd.append(bin_data['d'])
bd.append(bin_data2['d'])

plt.figure(1)
plt.subplot(3,1,1)
plt.hist(su0[0],color='r',normed=True,bins=20,label='init1')
plt.hist(su0[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[0],[0],[20],'k',label='u0')
plt.xlabel('u0')
plt.legend()
plt.subplot(3,1,2)
plt.hist(st0[0],color='r',normed=True,bins=20,label='init1')
plt.hist(st0[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[1],[0],[150],'k',label='t0')
# plt.xlabel('t0')
plt.legend()
plt.subplot(3,1,3)
plt.hist(ste[0],color='r',normed=True,bins=20,label='init1')
plt.hist(ste[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[2],[0],[100],'k',label='te')
# plt.xlabel('te')
plt.legend()

plt.figure(2)
plt.subplot(3,1,1)
plt.hist(bu0[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bu0[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[0],[0],[20],'k',label='u0')
# plt.xlabel('u0')
plt.legend()
plt.subplot(3,1,2)
plt.hist(bt0[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bt0[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[1],[0],[150],'k',label='t0')
# plt.xlabel('t0')
plt.legend()
plt.subplot(3,1,3)
plt.hist(bte[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bte[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[2],[0],[100],'k',label='t0')
# plt.xlabel('te')
plt.legend()
plt.figure(3)
plt.subplot(3,1,1)
plt.hist(bphi[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bphi[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[3],[0],[50],'k',label='phi')
# plt.xlabel('phi')
plt.legend()
plt.subplot(3,1,2)
plt.hist(bq[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bq[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[4],[0],[10],'k',label='q')
# plt.xlabel('q')
plt.legend()
plt.subplot(3,1,3)
plt.hist(bd[0],color='r',normed=True,bins=20,label='init1')
plt.hist(bd[1],color='b',normed=True,bins=20,label='init2')
plt.vlines(actualparams[5],[0],[10],'k',label='d')
# plt.xlabel('d')
plt.legend()
plt.show()
