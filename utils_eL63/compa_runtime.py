import numpy as np
import time

#-----------------------------------------
# general settings

verbose = True

ndim = 4

Nt = 1000
dt = 0.01

# initial condition
z0 = np.random.randn(ndim)


#-----------------------------------------
# new method
from eL63 import embeddedLorenz63
from integrators.integrate import integrate_runge_kutta

dic_LZ = {'ndim':ndim, 'sigma':10., 'beta':8./3., 'rho':28., 'kappa':1.}
eL63 = embeddedLorenz63(dic=dic_LZ)
f = eL63.create_tendencies()

start = time.time()
#
tt, traj, tend = integrate_runge_kutta(f, t0=0., t=(Nt-1)*dt, dt=dt, ic=z0)
z = traj.T
dz = tend.T
#
end = time.time()
elapsed = end - start
print('time = '+str(elapsed)+' s')
#
if verbose:
    print('------')    
    print(z[-1, :])
    print('------')
    print(dz)

