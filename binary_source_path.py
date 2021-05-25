import numpy as np
import matplotlib.pyplot as plt
import math

xs_binary = []
ys_binary = []

def source(t,t0,te,u0,phi):

    phi = math.radians(phi)
    xs_binary = np.array([(i-t0)/te*np.cos(phi)+u0*np.sin(phi) for i in t])
    ys_binary = np.array([u0*np.cos(phi)-((i-t0)/te)*np.sin(phi) for i in t])

    return xs_binary,ys_binary

if(0):
    plt.figure(1)
    plt.plot(source(t,t0,te,u0,phi)[0],source(t,t0,te,u0,phi)[1])
    plt.grid()
    plt.show()
