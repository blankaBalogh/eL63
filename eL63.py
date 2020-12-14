'''
"embedded" Lorenz'63 model (eL63) definition.
'''

import numpy as np
from scipy.linalg import qr
from data import orbits

class embeddedLorenz63():
    '''
    ndim : number of dimensions of eLM (embedded Lorenz63 model) : ndim >=3
    sigma, beta, rho : Lorenz63 parameters
    kappa : \dot{x}_{j}(t) = - \kappa x_{j}(t) for 3 <= j <= ndim
    
    aleamat : if False : identity matrix. Else : matrix from BZ to BX randomly 
            generated.
    matrix : transformation matrix
    invmat : inverse of matrix
    '''
    
    def __init__(self, dic=None):

        self.name = 'eL63'
        
        self.ndim = 3
        self.sigma = 10.
        self.beta = 8./3.
        self.rho = 28.
        self.kappa = 1.
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


    def Bz_to_Bx(self, z):
        '''
        '''
        z_2d = z.reshape(-1, self.ndim)
        x_2d = np.dot(self.matrix, z_2d.T).T
        return x_2d.reshape(z.shape)


    def Bx_to_Bz(self, x):
        '''
        '''
        x_2d = x.reshape(-1, self.ndim)
        z_2d = np.dot(self.invmat, x_2d.T).T
        return z_2d.reshape(x.shape)


    def f(self):
        '''
        '''
        aleamat = self.aleamat
        
        def func(x):
            '''
            '''
            if aleamat:
                z = self.Bx_to_Bz(x)
            else:
                z = x

            dz = self.f_Bz()(z)

            if aleamat:
                dx = self.Bz_to_Bx(dz)
            else:
                dx = dz

            return dx
        
        return func


    def f_Bz(self):
        '''
        '''
        ndim = self.ndim        
        sigma, rho, beta, kappa = self.sigma, self.rho, self.beta, self.kappa
        
        def func(z):
            '''
            '''
            assert z.shape[-1] == ndim, 'rightmost dimension of z must be equal to d = %s (dim of eL63)'%self.ndim

            if len(z.shape) == 1:
                z = z.reshape(1, *z.shape)
            
            dz = np.zeros_like(z)
            
            z_1, z_2, z_3, z_i = z[..., 0], z[..., 1], z[..., 2], z[..., 3:]
            
            # Lorenz'63 model
            dz[..., 0]  = sigma * (z_2 - z_1)
            dz[..., 1]  = rho * z_1 - z_2 - z_1 * z_3
            dz[..., 2]  = z_1 * z_2 - beta * z_3
            
            # restoring forces
            dz[..., 3:] = -kappa * z_i
            
            return dz

        return func


    def f_Bz_ty(self, shape):
        '''
        for use in ode_solvers : scipy.integrate.solve_ivp
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(t, y)
        '''
        def func(t, y):
            y = y.reshape(shape)
            result = self.f_Bz()(y)
            return result.reshape(-1)
        
        return func


    def f_Bz_yt(self, shape):
        '''
        for use in ode_solvers : scipy.integrate.odeint and formal_ode_solver
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(y, t)
        '''
        def func(y, t):
            y = y.reshape(shape)
            result = self.f_Bz()(y)
            return result.reshape(-1)
        
        return func


    def Bz_orbits(self, z0, n_steps=200, dt=0.01, solver='formal', method='RK4', x_bounds=None):
        '''
        
        Numerical integration of :
            dz/dt = f_eL63(z)
            z(t=0) = z0
            z in B_z
        
        z0          initial condtions : (n_orbits, ndim) or (ndim,)
        n_steps     number of steps/iterations for each orbit        
        dt          timestep

        '''
        if solver == 'solve_ivp':
            f = self.f_Bz_ty(z0.shape)
        else:
            f = self.f_Bz_yt(z0.shape)
        return orbits(f, z0, n_steps, dt, solver, method, x_bounds)


    def _orbits(self, x0, n_steps=200, dt=0.01, solver='formal', method='RK4', x_bounds=None):
        '''
        
        Numerical integration of eL63:
            dx/dt = f_eL63(x)
            x(t=0) = x0
            x in B_x
        
        x0          initial condtions : (n_orbits, ndim) or (ndim,)
        n_steps     number of steps/iterations for each orbit        
        dt          timestep

        '''
        if self.aleamat:
            z0 = self.Bx_to_Bz(x0)
        else:
            z0 = x0
        
        zt = self.Bz_orbits(z0, n_steps=n_steps, dt=dt, solver=solver, method=method, x_bounds=x_bounds)
        
        if self.aleamat:
            xt = self.Bz_to_Bx(zt)
        else:
            xt = zt
        return xt
