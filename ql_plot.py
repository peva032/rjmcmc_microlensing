import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ql_model_data.csv')
count = data['count']
count1 = data['count1']
count2 = data['count2']
mprob = data['model_probability']
model_index = data['model data']

mprob1 = [mprob[ind] for ind,i in enumerate(model_index) if i==1]
mprob2 = [mprob[ind] for ind,i in enumerate(model_index) if i==2]
mprob1_other = 1-np.array(mprob1)
mprob2_other = 1-np.array(mprob2)
classify1 = []
classify2 = []
model_count1 = 0
model_count2 = 0

for ii,jj in enumerate(count1):

    if jj>count2[ii] and model_index[ii]==1:
        classify1.append(1)
        model_count1=model_count1+1
    elif jj<count2[ii] and model_index[ii]==2:
        classify2.append(1)
        model_count2=model_count2+1
    elif jj>count2[ii] and model_index[ii]==2:
        classify2.append(0)
        model_count2=model_count2+1
    elif jj<count2[ii] and model_index[ii]==1:
        classify1.append(0)
        model_count1=model_count1+1
classify = [float(sum(classify1))/float(model_count1),float(sum(classify2))/float(model_count2)]
av_prob1 = [np.mean(mprob1),np.mean(mprob1_other)]
av_prob2 = [np.mean(mprob2_other),np.mean(mprob2)]
std_prob1 = [np.std(mprob1),np.std(mprob1_other)]
std_prob2 = [np.std(mprob2_other),np.std(mprob2)]

d = {'Classification accuracy':classify,'average prob m1':av_prob1,'std prob m1':std_prob1,'average prob m2':av_prob2,'std prob m2':std_prob2}
df1 = pd.DataFrame(d)
print(df1.head())
