import numpy as np
import matplotlib.pyplot as plt

u_0 = [0.1,0.2,0.3,0.4,0.5]
tau = np.linspace(-1,1,100)

plt.figure(2)
for ind,i in enumerate(u_0):
    u = np.array((tau**2+i**2)**0.5)
    A = (u**2 + 2)/(u*(u**2 + 4)**0.5)
    s = str(i)
    plt.plot(tau,A,label='u_0='+s)

plt.legend()
plt.title('Family of single lens light curves')
plt.xlabel('$(t-t_0)/t_E$')
plt.ylabel('$Magnification,\; A(u)$')
plt.show()
