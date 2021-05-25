import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt

#The aim of this program is to get a reversible jump mcmc to explore a two different gaussian models
#One with one mode and the other with two.

def model1(x1,mu1,icov1):
    diff1 = x1-mu1
    return -np.dot(diff1,np.dot(icov1,diff1))/2.0

def model2(x2,mu2,icov2):
    diff2 = x2-mu2
    return -np.dot(diff2,np.dot(icov2,diff2))/2.0
