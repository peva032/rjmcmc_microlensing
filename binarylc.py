import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
from binary_source_path import source

#-----------------------------------------------------------------------------
# This function generates the light curve for a binary lens.
#-----------------------------------------------------------------------------
def binary2(t,u0,t0,te,phi,q,d):

    zs_real = source(t,t0,te,u0,phi)[0]
    zs_imag = source(t,t0,te,u0,phi)[1]
    q = float(q)
    d = float(d)
    m2 = 1/(1+q)
    m1 = 1-m2
    z1 = d/abs(1+1/q)
    z2 = -z1/q
    mag = []
    for ind,i in enumerate(zs_real):
        zs = complex(i,zs_imag[ind])
        zs1 = zs.conjugate()-z1
        zs2 = zs.conjugate()-z2
        zsa = 2*zs.conjugate()-z1-z2
        zsm = zs1*zs2

# Calculating Coefficients of C(Z) a 5th order polynomial.
#-----------------------------------------------------------------------------
        a = -(z1+z2)
        b = z1*z2

        k0 = b**2
        k1 = 2*a*b
        k2 = ((a**2)+2*b)
        k3 = 2*a
        k4 = 1

        g0 = k0*zsm
        g1 = b*zsa + k1*zsm
        g2 = 1 + a*zsa + k2*zsm
        g3 = zsa + k3*zsm
        g4 = k4*zsm

        c1 = -g0*zs - k0*zs.conjugate()
        c2 = g0 -g1*zs -k1*zs.conjugate()-b
        c3 = g1 -g2*zs -k2*zs.conjugate()-a
        c4 = g2 -g3*zs -k3*zs.conjugate()-1
        c5 = g3 -k4*zs.conjugate()-g4*zs
        c6 = g4

# Solving the 5th order polynomial.
#-----------------------------------------------------------------------------
        c = [c6,c5,c4,c3,c2,c1]
        p = np.poly1d(c)
        z = np.roots(p)
        z_use = []

# Check to see which roots satisfy the lens equation
#-----------------------------------------------------------------------------
        for ii in z:
             check = zs - ii + (m1/(ii.conjugate()-z1.conjugate())) + (m2/(ii.conjugate()-z2.conjugate()))
             if(abs(check) <= 1e-10):
                 #This sets a threshold for the substitution check, to get around the computer error
                 z_use.append(ii)

# Using the solutions for image positions, calculate the magnitude.
#-----------------------------------------------------------------------------
        A = 0;
        for jj in z_use:
            # First calculating the two components of the determinant
            delzs_delzbar = jj.conjugate()/((jj.conjugate() - z1.conjugate())*(jj.conjugate() - z2.conjugate())**2)- 1/((jj.conjugate()-z1.conjugate())*(jj.conjugate() - z2.conjugate()))+ jj.conjugate()/(((jj.conjugate()-z1.conjugate())**2)*(jj.conjugate()-z2.conjugate()))

            delzsbar_delz = jj/((jj-z1)*(jj-z2)**2) - 1/((jj-z1)*(jj-z2)) - jj.conjugate()/(((jj-z1)**2)*(jj-z2))
            # Calculating the determinant of J
            det_J = 1 - delzs_delzbar*delzsbar_delz
            #Calculating the Amplitude of the ith image
            A_i = 1/abs(det_J)
            A = A+A_i
        mag.append(A)

    return mag
