# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:06:25 2024

@author: Tom
"""
# Clean Code
# Block 1 imports 

import warnings

# Specify the file path where the warnings originate
file_path = "C:/Users/Tom/AppData/Roaming/Python/Python311/site-packages/pysindy/optimizers/stlsq.py"

# Suppress all warnings originating from the specified file
warnings.filterwarnings("ignore", category=UserWarning, module=file_path)

import pysindy as ps

#from pysindy.utils import linear_damped_SHO
#from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
#from pysindy.utils import hopf
#from pysindy.utils import lorenz

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
#from scipy.io import loadmat


def SINDy_model(Threshold,initCondition=[2,0,1]):
    #return the model
    
    # Integrator keywords for solve_ivp
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-12

    dt = .01
    t_train = np.arange(0, 50, dt)
    t_train_span = (t_train[0], t_train[-1])
    
    #initial conditions
    x0_train = initCondition
    #Training Data generation
    x_train = solve_ivp(linear_3D, t_train_span,x0_train, t_eval=t_train, **integrator_keywords).y.T
    poly_order = 5
    
    try:
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=Threshold),
            feature_library=ps.PolynomialLibrary(degree=poly_order)
            )
        # Fit the model
        model.fit(x_train, t=dt)
    except UserWarning as warn:
        print('Some Sparcity happened and we are not happy')
    #
    
    modelerror = 1 
    ErrorTarget = (5/9000)
    
    itrlimit = 50
    itr = 0
    ndecimals = 5
    
    return model

def SINDyThresholdOptimization(model):
    
    a11 = model.coefficients()[0,1]
    a12 = model.coefficients()[1,1]
    a13 = model.coefficients()[2,1]
    a21 = model.coefficients()[0,2]
    a22 = model.coefficients()[1,2]
    a23 = model.coefficients()[2,2]
    a31 = model.coefficients()[0,3]
    a32 = model.coefficients()[1,3]
    a33 = model.coefficients()[2,3]

    modelerror = abs(a11-(-0.1))+abs(a12-(-2))+abs(a21-(2))+abs(a22-(-0.1))+abs(a33-(-0.3))
    
    return modelerror

def Plot2d_for_SINDy_model(t_train,x_train,x_sim):
    #we do not need this for now
    plot_kws = dict(linewidth=2)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_train[:, 0], x_train[:, 1], x_train[:, 2], "r", label="$x_k$", **plot_kws)
    ax.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], "k--", label="model", **plot_kws)
    ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
    ax.legend()
    fig.show()

def Plot3d_for_SINDy_model(t_train,x_train,x_sim):
    #we do not need this for now
    plot_kws = dict(linewidth=2)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
    plt.plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    plt.plot(t_train, x_train[:, 2], "g", label="$x_2$", **plot_kws)
    plt.plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
    plt.plot(t_train, x_sim[:, 1], "k--", **plot_kws)
    plt.plot(t_train, x_sim[:, 2], "k--", **plot_kws)
    plt.plot(t_train, abs(x_train-x_sim)[:,0]+abs(x_train-x_sim)[:,1]+abs(x_train-x_sim)[:,2], "--b",label="$Error$", **plot_kws)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("$x_k$")
    fig.show()
    
models = []
Tests = [0.3,0.03]
for eachtest in Tests:
    models.append(SINDy_model(eachtest, [2,0,1]))

for eachmodel in models:
    print(SINDyThresholdOptimization(eachmodel))
    eachmodel.print()