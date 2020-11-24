'''
"embedded" Lorenz'63 model (eL63) definition.
'''

import numpy as np
import sparse as sp
from scipy.linalg import qr
from numba import njit
from pyDOE import lhs

from integrate import integrate_runge_kutta

class embeddedLorenz63():
    '''
    ndim : number of dimensions of eLM (embedded Lorenz63 model) : ndim >=3
    sigma, beta, rho : Lorenz63 parameters
    kappa : \dot{x}_{j}(t) = - \kappa x_{j}(t) for 3 <= j <= ndim
    minval, maxval : model bounds for each component. 4th min/max val is associated 
            with minimal & maximal values for jth component, where j>3.

    aleamat : if False : identity matrix. Else : matrix from BZ to BX randomly 
            generated.
    matrix : transformation matrix
    invmat : inverse of matrix
    '''
    
    def __init__(self, dic=None):

        self.ndim = 3
        self.sigma = 10.
        self.beta = 8./3.
        self.rho = 28.
        self.kappa = 1.
        self.minval = [-20., -20.,  0., -5.]
        self.maxval = [ 20.,  20., 40.,  5.]
        self.aleamat = False
        self.set_params(dic)

        self.matrix = None
        self.invmat = None
        self.compute_matrix()


    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val


    def compute_matrix(self):
        '''
        '''
        if self.aleamat:
            H = np.random.randn(self.ndim, self.ndim)
            Q, R = qr(H)
            self.matrix = Q
        else:
            self.matrix = np.identity(self.ndim)
            
        self.invmat = np.linalg.inv(self.matrix)


    def BZ_to_BX(self, x):
        '''
        '''
        return np.dot(self.matrix, x.T).T


    def BX_to_BZ(self, y):
        '''
        '''
        return np.dot(self.invmat, y.T).T

    
    def tendencies(self):
        '''
        '''

        sigma, rho, beta, kappa = self.sigma, self.rho, self.beta, self.kappa

        def f(t, v):
            '''
            "embedded" Lorenz'63 model =
             (Lorenz'63 model + extra-dimensions : relaxations)
            '''

            singledim = (len(v.shape) == 1)
            
            reshapdim = (self.ndim not in v.shape)
            
            if singledim:
                v = v.reshape(1, -1)

            if reshapdim:
                v = np.reshape(v, (-1, self.ndim))
                
            dv = np.zeros_like(v)          
            
            x, y, z, ext = v[:,0], v[:,1], v[:,2], v[:,3:]
            
            dv[:, 0]  = sigma*(y-x)
            dv[:, 1]  = rho*x - y - x*z
            dv[:, 2]  = x*y - beta*z
            dv[:, 3:] = -kappa*ext

            if singledim == 1:
                dv = np.squeeze(dv)

            if reshapdim:
                dv = np.reshape(dv, (1, -1))

            return dv

        return f

    def trajectories(self, f, dt, n_traj=10, n_step=200, in_BX=False, z0=None):
        '''
        f       \dot{x} = f(t, x)
        dt      timestep
        n_traj  number of trajectories/orbits
        n_step  number of steps/iterations for each trajectory/orbit
        in_BX   initial condition in BX ?
        z0      initial condition in BZ
        '''
        if z0 is None:
            z00 = np.random.random((n_traj, self.ndim))
            zmin = np.r_[np.array(self.minval[0:3]), self.minval[3]*np.ones(self.ndim-3)]
            zmax = np.r_[np.array(self.maxval[0:3]), self.maxval[3]*np.ones(self.ndim-3)]
            z0 = (zmax - zmin)*z00 +zxmin
            x0 = self.BZ_to_BX(z0)

        else:
            if in_BX:
                x0 = z0

        if in_BX:
            ic = x0.reshape(1, -1)
        else:
            ic = z0.reshape(1, -1)

        tt, traj, tend = integrate_runge_kutta(f, t0=0., t=(n_step-1)*dt, dt=dt, ic=ic)

        if n_traj > 1:
            x_t = np.reshape(traj, (n_traj, self.ndim, -1))
            dx_t = np.reshape(tend, (n_traj, self.ndim, -1))
            x_, dx_ = x_t[:, :, -1], dx_t[:, :, -1] # -- save last iteration
        else:
            x_ = traj.T
            dx_ = tend.T

        if in_BX:
            z = self.BX_to_BZ(x_)
            dz = self.BX_to_BZ(dx_)
        else:
            z, dz = x_, dx_
            x = self.BZ_to_BX(z)
            dx = self.BZ_to_BX(dz)

        return x, dx, z, dz

    
    def generate_LHS_sampling(self, x, n_samples=100):
        '''
        x is an eL63 trajectory (in BX)
        '''
        
        # Getting sample boundaries
        min_x, max_x = np.min(x, axis=0), np.max(x, axis=0)
        
        # Latin Hypercube sampling : in BX
        lhs_x = (lhs(self.ndim, samples=n_samples, criterion="center"))*(np.array(max_x)-np.array(min_x)) + np.array(min_x)
        
        # Transferring LHS from BX to BZ
        lhs_z = self.BX_to_BZ(lhs_x)

        # Getting time derivatives for LHS in BZ (lhs_z)
        f = self.tendencies()
        lhs_dz = np.zeros_like(lhs_z)
        for t in range(lhs_z.shape[0]):
            lhs_dz[t, :] = f(0., lhs_z[t, :])
        
        # Transferring dx from BZ to BX
        lhs_dx = self.BZ_to_BX(lhs_dz)
        
        return lhs_x, lhs_dx


if __name__ == '__main__':

    dic_LZ = {'ndim':4, 'sigma':10., 'beta':8./3., 'rho':28., 'kappa':1.}
    eL63 = embeddedLorenz63(dic=dic_LZ)

    x = np.random.randn(eL63.ndim)

    z = eL63.BX_to_BZ(x)
    xx = eL63.BZ_to_BX(z)

    print(x)
    print(xx)

    from integrators.integrate import integrate_runge_kutta

    ic = np.array([10., 10., 0])
    
    f = eL63.create_tendencies()
    tt, traj, tend = integrate_runge_kutta(f, t0=0., t=0.02, dt=0.01, ic=ic)

    print(traj)
    print(tend)
    
    from utils.utils_dimN import *

    par_LZ__ = [10, 28, 8/3]
    target_dim = 3
    z0 = (np.random.random(target_dim)-0.5)*10
    z0[:3] = [10, 10, 0]
    N  = 3
    dt = 0.01
    int_scheme = "RK4"
    tau = 1
    
    Z_LZ, dZ_LZ = generate_trajectory_Q(dt, N, x0, tau, par_LZ=par_LZ__, onlyLZ=True, scheme='RK4', verbose=0)
    print(np.shape(Z_LZ))
    print(Z_LZ[-1, :])
    print(dZ_LZ)
