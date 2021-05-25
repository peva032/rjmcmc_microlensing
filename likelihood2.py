import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from binarylc import binary2
from numpy.linalg import inv

def like(t,y,thet):
    sigmae = 0.255
    cov = (sigmae**2)*np.identity(len(y))
    Ax = binary2(t,thet[0],thet[1],thet[2],thet[3],thet[4],thet[5])
    pot = np.array(y).reshape(len(y),1)-np.array(Ax).reshape(len(y),1)
    like = np.exp(-np.dot(np.dot(np.transpose(pot),inv(cov)),pot)/2.0)
    return like[0][0]


theta1 = np.linspace(0,0.02,100)
data = pd.read_csv('binary_data.csv')
t = data['t']
y = data['A']
like1 = []

for i  in theta1:

    like1.append(like(t,y,[i,15,10,10,0.001,0.6]))

plt.figure()
plt.plot(theta1,like1)
plt.show()
