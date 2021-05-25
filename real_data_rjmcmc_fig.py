import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from rjfun import *

pwd = os.getcwd()
mi = []
su0 = []
st0 = []
ste = []
bu0 = []
bt0 = []
bte =[]
bphi = []
bq = []
bd = []
table = []
table2 = []
# t = []
# A = []


sin_data = pd.read_csv(pwd+'/outputs/realdata/output_(single).csv')
bin_data = pd.read_csv(pwd+'/outputs/realdata/output_(binary).csv')
mi_data = pd.read_csv(pwd+'/outputs/realdata/mtrace.csv')

data = pd.read_csv('OB110251.csv',sep='\t')
data.columns = ['t','A','sigmaA']
t = data['t']
A = data['A']
sigmaA = data['sigmaA']
print(t[0])
t  = t - t[0]
t = np.array(t[1400:3450])
y = np.array(A[1400:3450])
sigmaA = sigmaA[1400:3450]
mi.append(mi_data['m'])
su0.append(sin_data['u0'])
st0.append(sin_data['t0'])
ste.append(sin_data['te'])
bu0.append(bin_data['u0'])
bt0.append(bin_data['t0'])
bte.append(bin_data['te'])
bphi.append(bin_data['phi'])
bq.append(bin_data['q'])
bd.append(bin_data['d'])
# t.append(data['t'])
# A.append(data['A'])

count1 = 0
count2 = 0
for ii in mi_data['m']:
    if(ii==1):
        count1 = count1+1
    if(ii==2):
        count2 = count2+1

Nt = float(len(mi_data['m']))
var = ((count1/Nt)*(count2/Nt))/Nt
table=np.array([count1/Nt,count2/Nt,var]).reshape(1,3)
table2=np.array([np.mean(bu0[0]),np.std(bu0[0]),np.mean(bt0[0]),np.std(bt0[0]),np.mean(bte[0])\
,np.std(bte[0]),np.mean(bphi[0]),np.std(bphi[0]),np.mean(bq[0]),np.std(bq[0]),np.mean(bd[0]),np.std(bd[0])]).reshape(1,12)


table = pd.DataFrame(table,columns=['P_1','P_2','std'])
table2 = np.array(table2).reshape(1,12)
table2 = pd.DataFrame(table2,columns=['u0','sig-u0','t0','sig-t0','te','sig-te','phi','sig-phi','q','sig-q','d','sig-d'])
table.to_csv('real_data_model_prob.csv',index=False)
table2.to_csv('real_data_bin_estimates.csv',index=False)

t_plot = np.linspace(0,700,2000)
tt = np.linspace(0,700,2000)

plt.figure(2)
# plt.errorbar(t,A,'kx',markersize=0.7,label='OB110251_data')
plt.errorbar(t,y,sigmaA,fmt='ko',markersize=1,label='OB110251_data')
plt.plot(t_plot,MT(t_plot,np.mean(su0[0]),np.mean(st0[0]),np.mean(ste[0])),'b--',linewidth=1,label='m1')
plt.plot(t_plot,binary2(t_plot,np.mean(bu0[0]),np.mean(bt0[0]),np.mean(bte[0]),np.mean(bphi[0]),np.mean(bq[0]),np.mean(bd[0])),'r--',linewidth=1,label='m2')
plt.xlabel('time')
plt.ylabel('magnification')
plt.legend()
plt.figure(3)
plt.subplot(1,2,1)
plt.hist(mi[0],bins=5)
# plt.legend()
plt.xlabel('Model')
plt.subplot(1,2,2)
plt.plot(np.linspace(0,len(mi[0]),len(mi[0])),mi[0],'b-',linewidth=0.5,label='m-trace')
plt.legend()
plt.plot(np.linspace(0,len(mi[0]),len(mi[0])),mi[0],'ko',markersize=0.7)
plt.xlabel('iterations')
plt.ylabel('Model samples')
plt.show()
# plt.subplot(3,3,4)
# plt.plot(t[1],A[1],'kx',markersize=0.7,label='(2)')
# plt.plot(t_plot,MT(t_plot,np.mean(su0[1]),np.mean(st0[1]),np.mean(ste[1])),'b--',linewidth=1,label='m1')
# plt.plot(t_plot,binary2(t_plot,np.mean(bu0[1]),np.mean(bt0[1]),np.mean(bte[1]),np.mean(bphi[1]),np.mean(bq[1]),np.mean(bd[1])),'r--',linewidth=1,label='m2')
# plt.legend()
# plt.subplot(3,3,5)
# plt.hist(mi[1],bins=5,label='(2)')
# plt.legend()
# plt.xlabel('Model')
# plt.subplot(3,3,6)
# plt.plot(np.linspace(0,len(mi[1]),len(mi[1])),mi[1],'b-',linewidth=0.5,label='m-trace')
# plt.legend()
# plt.plot(np.linspace(0,len(mi[1]),len(mi[1])),mi[1],'ko',markersize=0.7)
# plt.subplot(3,3,7)
# plt.plot(t[2],A[2],'kx',markersize=0.7,label='(3)')
# plt.plot(t_plot,MT(t_plot,np.mean(su0[2]),np.mean(st0[2]),np.mean(ste[2])),'b--',linewidth=1,label='m1')
# plt.plot(t_plot,binary2(t_plot,np.mean(bu0[2]),np.mean(bt0[2]),np.mean(bte[2]),np.mean(bphi[2]),np.mean(bq[2]),np.mean(bd[2])),'r--',linewidth=1,label='m2')
# plt.legend()
# plt.subplot(3,3,8)
# plt.hist(mi[2],bins=5,label='(3)')
# plt.legend()
# plt.xlabel('Model')
# plt.subplot(3,3,9)
# plt.plot(np.linspace(0,len(mi[2]),len(mi[2])),mi[2],'b-',linewidth=0.5,label='m-trace')
# plt.legend()
# plt.plot(np.linspace(0,len(mi[2]),len(mi[2])),mi[2],'ko',markersize=0.7)
# plt.show()
