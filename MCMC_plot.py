import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def mcmcplot(ax,M,a,b,c):

    if M==1:
        col = 'g'
    elif M==2:
        col = 'b'
    ax.scatter(a, b, c,c=col)
    plt.title('Parameter Space of 2-Model RJMCMC')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('b')
    plt.pause(0.1)
