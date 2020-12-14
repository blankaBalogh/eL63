import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time

from scipy.integrate import solve_ivp, odeint
from pyDOE import lhs


def f_ty(dynamical_model, shape):
    '''
    for use in ode_solvers : scipy.integrate.solve_ivp
        (i) transform 1-dim input to n-dim input
        (ii) transform (y, t)->f(y) to (y, t)->f(t, y)
    '''
    def func(t, y):
        y = y.reshape(-1, shape[1])
        result = dynamical_model.f()(y)
        return result.reshape(-1)
        
    return func


def f_yt(dynamical_model, shape):
    '''
    for use in ode_solvers : scipy.integrate.odeint and formal_ode_solver
        (i) transform 1-dim input to n-dim input
        (ii) transform (y, t)->f(y) to (y, t)->f(y, t)
    '''
    def func(y, t):
        y = y.reshape(-1, shape[1])
        result = dynamical_model.f()(y)
        return result.reshape(-1)
        
    return func


def formal_ode_solver(f, x0, time, method='RK4', x_bounds=None, verbose=False):
    '''
    
    Numerical integration of:
        dx / dt = f(t, x) or f(x, t)
        x(t=0) = x0

    - method == 'Euler' : explicit (Euler) integration
    - method == 'RK4' : fourth-order Runge-Kutta scheme
    
    x_bounds = (xmin, xmax, freq_chk_bounds); default : None
    freq_chk_bounds (unit : dt) : check_bounds (xmin/xmax) frequency   

    '''
    solution = np.zeros([time.size, x0.size])
    solution[0, :] = x0
    sTime = time.size - 1

    if x_bounds is None:
        freq_chk_bounds = sTime + 1
    else:
        xmin, xmax, freq_chk_bounds = x_bounds
        i_last_check = 1
    
    # indexes of stable orbits
    ista = np.ones(*x0.shape, dtype=bool)
    
    def stable_orbits(sol_, xmin, xmax, verbose=False):
        '''
        '''
        # sol_n.shape : number of timesteps, number of orbits, number of model dimension (*xmin.shape)
        sol_n = sol_.reshape((sol_.shape[0], -1, *xmin.shape))
        imin = np.sum(sol_n < xmin, axis=(0, 2))
        imax = np.sum(sol_n > xmax, axis=(0, 2))
        # ista_ : indexes of 'new' stable orbits
        ista_ = ( imin + imax == 0 )
        # istab : indexes of 'new' stable orbits (*ndim)
        istab = np.repeat(ista_, *xmin.shape)
        # print list of 'new' unstable orbits
        if verbose and any(ista_ == 0):
            indexes = np.arange(len(ista_))
            print(' > unstable solutions for orbits: ', indexes[ista_==0])
        return istab
    
    if method == 'RK4':
        for i in range(sTime):
            h = time[i+1] - time[i]
            k1 = f(solution[i, ista], time[i])
            k2 = f(solution[i, ista] + 0.5 * h * k1, time[i] + 0.5*h)
            k3 = f(solution[i, ista] + 0.5 * h * k2, time[i] + 0.5*h)
            k4 = f(solution[i, ista] + h * k3 , time[i+1])
            solution[i+1, ista] = solution[i, ista] + h * ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6.
            if (i+1)%freq_chk_bounds == 0:
                if verbose:
                    print('check bounds @timestep %6d'%(i+1))
                ista = ista & stable_orbits(solution[i_last_check:i+2, :], xmin, xmax, verbose=verbose)
                solution[i+2:, ~ista] = np.nan
                i_last_check += freq_chk_bounds
                nsta = np.sum(ista) / len(xmin)
                if verbose:
                    print('number of stable orbits = %4d'%nsta)
                if nsta == 0:
                    print('> stop integration @timestep %6d for method %s'%(i+1, method))
                    return solution
        return solution

    elif method == 'Euler':
        for i in range(sTime):
            h = time[i+1] - time[i]
            solution[i+1,:] = solution[i,:] + h * f(solution[i,:], time[i])
            if (i+1)%freq_chk_bounds == 0:
                if verbose:
                    print('check bounds @timestep %6d'%(i+1))
                ista = ista & stable_orbits(solution[i_last_check:i+2, :], xmin, xmax, verbose=verbose)
                solution[i+2:, ~ista] = np.nan
                i_last_check += freq_chk_bounds
                nsta = np.sum(ista) / len(xmin)
                if verbose:
                    print('number of stable orbits = %4d'%nsta)
                if nsta == 0:
                    print('> stop integration @timestep %6d for method %s'%(i+1, method))
                    return solution
        return solution
    
    else:
        print('formal_ode_solver: solver_method %s not known'%method)        
        return solution


def orbits(f, x0, n_steps=200, dt=0.01, solver='formal', method='RK4', x_bounds=None):
    '''
    
    Numerical integration of:
        dx / dt = f(t, x) or f(x, t)
        x(t=0) = x0

    Use solver (default:'formal') and method (default:'RK4') for integration

    - solver == 'formal' : use function formal_ode_solver
        - method == 'Euler' : explicit (Euler) integration
        - method == 'RK4' : fourth-order Runge-Kutta scheme
    - solver == 'odeint' : use scipy.integrate.odeint
    - solver == 'solve_ivp' : use scipy.integrate.solve_ivp
        - method == 'RK45_p' : method == 'RK45' and rtol = atol = 1.49012e-8
        - method == 'LSODA_p' : method == 'LSODA' and rtol = atol = 1.49012e-8
    
    x0          initial condtions : (n_orbits, ndim) or (ndim,)
    n_steps     number of steps/iterations for each orbit        
    dt          timestep

    x_bounds = (xmin, xmax, freq_chk_bounds); default : None
    freq_chk_bounds (unit : dt) : check_bounds (xmin/xmax) frequency

    Returns: a numpy.ndarray sol(n_steps, *x0.shape)

    '''
    ishp = x0.shape # initial shape
    x0_1d = x0.reshape(-1)
    
    t_star = np.arange(0., n_steps*dt, dt)

    if x_bounds is not None:
        xmin, xmax, freq_chk_bounds = x_bounds
        x_bounds = xmin, xmax, freq_chk_bounds

    if solver == 'formal':
        sol = formal_ode_solver(f, x0_1d, t_star, method, x_bounds)
        sol = sol.reshape(-1, *ishp)

    elif solver == 'odeint':
        sol = odeint(f, x0_1d, t_star)
        sol = sol.reshape(-1, *ishp)

    elif solver == 'solve_ivp':
        if method == 'RK45_p':
            sol = solve_ivp(f, t_span=[t_star.min(), t_star.max()], y0=x0_1d, t_eval=t_star, method='RK45', rtol=1.49012e-8, atol=1.49012e-8)
        elif method == 'LSODA_p':
            sol = solve_ivp(f, t_span=[t_star.min(), t_star.max()], y0=x0_1d, t_eval=t_star, method='LSODA', rtol=1.49012e-8, atol=1.49012e-8)            
        else:
            sol = solve_ivp(f, t_span=[t_star.min(), t_star.max()], y0=x0_1d, t_eval=t_star, method=method)
        sol = sol.y.T.reshape(-1, *ishp)

    else:
        print('orbits: solver %s not known'%solver)
        sol = None

    return sol


def generate_data(dynamical_model, x0, n_steps=100, dt=0.01, \
        compute_y=True, normalization=False, solver='formal', method='RK4', x_bounds=None):
    '''

    Numerical integration of:
        dx / dt = f(t, x) or f(x, t)
        x(t=0) = x0

    Use solver (default:'formal') and method (default:'RK4') for integration 
    
    The object 'dynamical_model' should have at least the attribute:
        dynamical_model.f() : callable, x \mapsto f(x), with x : n-dimensional
            used to compute f(x(t)) given x(t)
            
    dynamical_model.f() is transformed to : f(t, x) or f(x, t) with x : 1-dimensional
        using f_ty or f_yt

    x0          initial condtions : (n_orbits, ndim) or (ndim,)
    n_steps     number of steps/iterations for each orbit        
    dt          timestep

    x_bounds = (xmin, xmax, freq_chk_bounds); default : None
    freq_chk_bounds (unit : dt) : check_bounds (xmin/xmax) frequency

    Returns: a dictionary 'output' with
        - output['x'] : (n_steps, *x0.shape)
        - output['y'] : (n_steps, *x0.shape) = f(x) if compute_y = True
        - normalized values if normalization = True
            output['x_norm'] and output['y_norm']
            and the corresponding norms : output['norms'] = np.array([mean_x, mean_y, std_x, std_y])

    '''
    output = {}

    if len(x0.shape) == 1:
        x0 = x0.reshape(1, *x0.shape)
    
    if hasattr(dynamical_model, '_orbits'):
        xt = dynamical_model._orbits(x0, n_steps=n_steps, dt=dt, solver=solver, method=method, x_bounds=x_bounds)
    else:
        if solver == 'solve_ivp':
            f = f_ty(dynamical_model, x0.shape)
        else:
            f = f_yt(dynamical_model, x0.shape)
        xt = orbits(f, x0, n_steps=n_steps, dt=dt, solver=solver, method=method, x_bounds=x_bounds)

    output['x'] = xt

    if compute_y:
        yt = dynamical_model.f()(xt)
        output['y'] = yt
        
        if normalization:
            x_n, mean_x, std_x = normalize(xt)
            y_n, mean_y, std_y = normalize(yt)
            norms_ = np.array([mean_x, mean_y, std_x, std_y])
            output['x_norm'] = x_n
            output['y_norm'] = y_n
            output['norms'] = norms_

    return output


def generate_data_solvers(lst_solvers, dynamical_model, x_0, n_steps=100, dt=0.01, x_bounds=None, elapsed_time=False):
    '''

    Execute the function 'generate_data' for a list of (solver, method)

    Returns:
        a numpy.ndarray (n_solvers, n_steps, *x_0.shape)

    '''
    n_solvers = len(lst_solvers)
    xt = np.zeros([n_solvers, n_steps, *x_0.shape])

    if elapsed_time:
        elapsed_t = np.zeros([n_solvers])

    i = 0 
    for solver_, method_ in lst_solvers:
        if elapsed_time:
            start_t = time.time()
        output = generate_data(dynamical_model, x_0, n_steps=n_steps, dt=dt, \
                compute_y=False, solver=solver_, method=method_, x_bounds=x_bounds)
        if elapsed_time:
            end_t = time.time()
            elapsed_t[i] = end_t - start_t 
        xt[i, ...] = output['x']
        i += 1
    
    if elapsed_time:
        return xt, elapsed_t
    else:
        return xt


def save_orbits(xt_, lst_solvers, model_name, eL63_model=None, elapsed_times=None):
    '''
    '''
    df_ = pd.DataFrame()
    
    assert len(xt_.shape) == 4, 'len(xt.shape) must be equal to 4'
    n_solvers, n_steps, n_orbits, n_dim = xt_.shape

    if eL63_model is not None:
        zt_ = eL63_model.Bx_to_Bz(xt_)

    for i in range(n_solvers):
        solver, method = lst_solvers[i]
        # save only 4 components...
        for j in range(np.min([n_dim, 4])):
            for k in range(n_orbits):
                if eL63_model is not None:
                    xdata = np.zeros([n_steps, 2])
                    if np.isnan(np.sum(xt_[i, :, k, j])):
                        xt_[i, :, k, j] = np.nan
                        zt_[i, :, k, j] = np.nan
                    xdata[..., 0] = xt_[i, :, k, j]
                    xdata[..., 1] = zt_[i, :, k, j]
                    df_ijk = pd.DataFrame(xdata, columns=['x', 'z'])
                    df_ijk['timestep'] = np.arange(len(xdata))
                else:
                    df_ijk = pd.DataFrame(xt_[i, :, k, j], columns=['x'])

                df_ijk['model'] = model_name+'_d'+str(n_dim)
                df_ijk['solver'] = solver[0]+'_'+method
                df_ijk['i'] = j+1
                df_ijk['orbit'] = k+1
                if elapsed_times is not None:
                    df_ijk['elapsed_t'] = elapsed_times[i]
                df_ = pd.concat([df_, df_ijk])

    return df_


def generate_LHS(dynamical_model, xmin, xmax, n_samples=100, normalization=True):
    '''
    '''
    lhs_x = (lhs(dynamical_model.ndim, samples=n_samples, criterion='center'))*(np.array(xmax)-np.array(xmin)) + np.array(xmin)
    lhs_y = dynamical_model.f()(lhs_x)
    
    if normalization:
        x_n, mean_x, std_x = normalize(lhs_x)
        y_n, mean_y, std_y = normalize(lhs_y)
        norms_ = np.array([mean_x, mean_y, std_x, std_y])
        return lhs_x, lhs_y, x_n, y_n, norms_
    else:
        return lhs_x, lhs_y


def normalize(x):
    mean_ = np.mean(x, axis=tuple(range(0, x.ndim-1)))
    std_ = np.std(x, axis=tuple(range(0, x.ndim-1)))
    std_ = np.where(std_ == 0, 1.0, std_)
    x_n = (x - mean_) / std_
    return x_n, mean_, std_


def plot_orbits_distribution(df_, title, figname, figtype, palette='rocket', xaxis='i', yaxis='z', hue='orbit', col='model'):
    '''
    '''
    plt.figure()
    sns.set_theme(style='ticks', palette=palette)
    if col is None:
        sns_plot = sns.barplot(data=df_, x=xaxis, y=yaxis, hue=hue)         
    else:
        sns_plot = sns.catplot(data=df_, x=xaxis, y=yaxis, hue=hue, col=col, kind='box', showfliers=False)
    plt.savefig(figname+'.'+figtype)
