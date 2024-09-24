# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:53:46 2024

@author: Tom
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz

import pysindy as ps

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

dt = .01
t_train = np.arange(0, 50, dt)
t_train_span = (t_train[0], t_train[-1])



ListofInitialConditions = [[2,0,1],[7,7,7]]
for eachInitialCondtion in ListofInitialConditions:
    x0_train = eachInitialCondtion
    #Training Data generation
    x_train = solve_ivp(linear_3D, t_train_span,x0_train, t_eval=t_train, **integrator_keywords).y.T
    poly_order = 5
    
    TestThreshold = 0.0001  # should start with 0.02, 0.03, see the direction or pattern; it has a connection with line 62
    modelerror = 1 
    ErrorTarget = (5/9000)
    itrlimit = 50
    itr = 0
    ndecimals = 2
    errors = [1]
    
    found_target = False 
    
    for i in range(10**(ndecimals-1),10**ndecimals):
        for j in range(1,10):
            deci = i / (10**ndecimals) + j/(10** (ndecimals + 1))
        
            TestThreshold = deci  # should start with 0.02, 0.03, see the direction or pattern; it has a connection with line 62
            
            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=TestThreshold),
                feature_library=ps.PolynomialLibrary(degree=poly_order)
                )
            
            model.fit(x_train, t=dt)       
            # model.print()
            # line 60 should change to the conventional notation for each entry
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
            errors.append(modelerror)
            
            if modelerror < ErrorTarget:
                print('here')
                found_target = True
                break
            
            if errors[len(errors)-1] < modelerror:
                print(deci)
                found_target = True
                break 
            
        if found_target:
            break