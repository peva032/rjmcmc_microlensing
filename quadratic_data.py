import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)
def quad_data():
    sigmal = 1
    sigmae = 0.1
    N=100

    a = np.random.normal(0,sigmal)
    b = np.random.normal(0,sigmal)
    c = np.random.normal(0,sigmal)

    x = np.random.normal(0,sigmal,N).reshape(100)
    A = [x**2,x,np.ones(N)]
    e = np.random.normal(0,sigmae,N)
    y = [c*A[0]+b*A[1]+a*A[2]] + e
    y = y.reshape(100)

    d = {'x':x,'y':y}
    df = pd.DataFrame(d)
    df.to_csv('quadratic_data.csv',index=False)

    return a,b,c
