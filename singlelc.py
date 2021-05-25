import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MT(t,u0,t0,te):
    mt = ((u0**2) + (((t-t0)/te)**2) + 2)/(np.sqrt((u0**2)+((t-t0)/te)**2)*np.sqrt(((u0**2) + ((t-t0)/te)**2)+4))
    return mt
