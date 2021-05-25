import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from rjfun import *

pwd = os.getcwd()
q = [0.3,0.5,0.7]
N = 650
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
t = []
A = []
if(1):
    for ind,i in enumerate(q):
        data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_'+str(ind)+'_data_0.100_15.000_10.000_1.047_0.010_'+str(i)+'00_0.400.csv')
        sin_data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_'+str(ind)+'_output_(single).csv')
        bin_data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_'+str(ind)+'_output_(binary).csv')
        mi_data = pd.read_csv(pwd+'/outputs/singletobinary/fig1_singletobinary_'+str(ind)+'_mtrace.csv')

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
        t.append(data['t'])
        A.append(data['A'])

        count1 = 0
        count2 = 0
        for ii in mi_data['m']:
            if(ii==1):
                count1 = count1+1
            if(ii==2):
                count2 = count2+1

        # print(count1,count2)

        Nt = float(len(mi_data['m']))
        var = ((count1/Nt)*(count2/Nt))/Nt
        print(count1/Nt)
        print(count2/Nt)
        table.append([i,count1/Nt,count2/Nt,var])
        table2.append([i,np.mean(bu0[ind]),np.std(bu0[ind]),np.mean(bt0[ind]),np.std(bt0[ind]),np.mean(bte[ind])\
        ,np.std(bte[ind]),np.mean(bphi[ind]),np.std(bphi[ind]),np.mean(bq[ind]),np.std(bq[ind]),np.mean(bd[ind]),np.std(bd[ind])])

    table = np.array(table).reshape(len(table),4)
    table = pd.DataFrame(table,columns=['q','P_1','P_2','std'])
    table2 = np.array(table2).reshape(len(table2),13)
    table2 = pd.DataFrame(table2,columns=['q','u0','sig-u0','t0','sig-t0','te','sig-te','phi','sig-phi','q','sig-q','d','sig-d'])
    table.to_csv('single_to_binary_model_test_probabilities_fig1.csv',index=False)
    table2.to_csv('single_to_binary_parameter_estimates_fig1.csv',index=False)

    t_plot = np.linspace(0,30,300)
    tt = np.linspace(10,20,300)
    plt.figure(1)
    plt.plot(tt,binary2(tt,0.1,15,10,1.047,0.01,0.3),label='(1)')
    plt.plot(tt,binary2(tt,0.1,15,10,1.047,0.01,0.5),label='(2)')
    plt.plot(tt,binary2(tt,0.1,15,10,1.047,0.01,0.7),label='(3)')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('magnification')

    plt.figure(2)
    plt.subplot(3,3,1)
    plt.plot(t[0],A[0],'kx',markersize=0.7,label='(1)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[0]),np.mean(st0[0]),np.mean(ste[0])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[0]),np.mean(bt0[0]),np.mean(bte[0]),np.mean(bphi[0]),np.mean(bq[0]),np.mean(bd[0])),'r--',linewidth=1,label='m2')
    plt.legend()
    plt.subplot(3,3,2)
    plt.hist(mi[0],bins=5,normed=1,label='(1)')
    plt.legend()
    plt.xlabel('Model')
    plt.subplot(3,3,3)
    plt.plot(np.linspace(0,len(mi[0]),len(mi[0])),mi[0],'b-',linewidth=0.5,label='m-trace')
    plt.legend()
    plt.plot(np.linspace(0,len(mi[0]),len(mi[0])),mi[0],'ko',markersize=0.7)
    plt.subplot(3,3,4)
    plt.plot(t[1],A[1],'kx',markersize=0.7,label='(2)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[1]),np.mean(st0[1]),np.mean(ste[1])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[1]),np.mean(bt0[1]),np.mean(bte[1]),np.mean(bphi[1]),np.mean(bq[1]),np.mean(bd[1])),'r--',linewidth=1,label='m2')
    plt.legend()
    plt.subplot(3,3,5)
    plt.hist(mi[1],bins=5,normed=1,label='(2)')
    plt.legend()
    plt.xlabel('Model')
    plt.subplot(3,3,6)
    plt.plot(np.linspace(0,len(mi[1]),len(mi[1])),mi[1],'b-',linewidth=0.5,label='m-trace')
    plt.legend()
    plt.plot(np.linspace(0,len(mi[1]),len(mi[1])),mi[1],'ko',markersize=0.7)
    plt.subplot(3,3,7)
    plt.plot(t[2],A[2],'kx',markersize=0.7,label='(3)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[2]),np.mean(st0[2]),np.mean(ste[2])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[2]),np.mean(bt0[2]),np.mean(bte[2]),np.mean(bphi[2]),np.mean(bq[2]),np.mean(bd[2])),'r--',linewidth=1,label='m2')
    plt.legend()
    plt.subplot(3,3,8)
    plt.hist(mi[2],bins=5,normed=1,label='(3)')
    plt.legend()
    plt.xlabel('Model')
    plt.subplot(3,3,9)
    plt.plot(np.linspace(0,len(mi[2]),len(mi[2])),mi[2],'b-',linewidth=0.5,label='m-trace')
    plt.legend()
    plt.plot(np.linspace(0,len(mi[2]),len(mi[2])),mi[2],'ko',markersize=0.7)

    plt.figure(3)
    plt.subplot(1,3,1)
    plt.plot(t[0],A[0],'kx',markersize=0.7,label='(1)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[0]),np.mean(st0[0]),np.mean(ste[0])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[0]),np.mean(bt0[0]),np.mean(bte[0]),np.mean(bphi[0]),np.mean(bq[0]),np.mean(bd[0])),'r--',linewidth=1,label='m2')
    plt.axis([10,20,6,12])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('magnification')
    plt.subplot(1,3,2)
    plt.plot(t[1],A[1],'kx',markersize=0.7,label='(2)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[1]),np.mean(st0[1]),np.mean(ste[1])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[1]),np.mean(bt0[1]),np.mean(bte[1]),np.mean(bphi[1]),np.mean(bq[1]),np.mean(bd[1])),'r--',linewidth=1,label='m2')
    plt.axis([10,20,6,12])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('magnification')
    plt.subplot(1,3,3)
    plt.plot(t[2],A[2],'kx',markersize=0.7,label='(3)')
    plt.plot(t_plot,MT(t_plot,np.mean(su0[2]),np.mean(st0[2]),np.mean(ste[2])),'b--',linewidth=1,label='m1')
    plt.plot(t_plot,binary2(t_plot,np.mean(bu0[2]),np.mean(bt0[2]),np.mean(bte[2]),np.mean(bphi[2]),np.mean(bq[2]),np.mean(bd[2])),'r--',linewidth=1,label='m2')
    plt.axis([10,20,6,12])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('magnification')
    plt.show()
