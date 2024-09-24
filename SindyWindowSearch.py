# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:32:57 2024

@author: Tom
"""
import pysindy as ps
from decimal import *


#from pysindy.utils import linear_damped_SHO
#from pysindy.utils import cubic_damped_SHO
#from pysindy.utils import linear_3D
#from pysindy.utils import hopf
#from pysindy.utils import lorenz

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
#from scipy.io import loadmat


def linear_3D(t, x):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1], 1 * x[1] -0.3 * x[2]]


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
   
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=Threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order)
        )
    # Fit the model
    model.fit(x_train, t=dt)
   
    return model


def SINDyThresholdOptimization(model):
   
    #a11 = model.coefficients()[0,1]
    #a12 = model.coefficients()[1,1]
    #a13 = model.coefficients()[2,1]
    #a21 = model.coefficients()[0,2]
    #a22 = model.coefficients()[1,2]
    #a23 = model.coefficients()[2,2]
    #a31 = model.coefficients()[0,3]
    #a32 = model.coefficients()[1,3]
    #a33 = model.coefficients()[2,3]
    
    linear_matrix = np.array(model.coefficients()[0:3,1:4])
    actual_matrix = np.array([[-0.1,2,0],[-2,-0.1,0],[0,1,-0.3]])
    non_linear_matrix = np.array(model.coefficients())
    non_linear_matrix[0:3,1:4] = 0
    modelerrorLinear = np.sum(np.square(np.abs(linear_matrix - actual_matrix)))
    modelerrorNonLinear = np.sum(np.square( non_linear_matrix ))
    #modelerrorLinear = abs(a11-(-0.1))**2+abs(a12-(-2))**2+abs(a21-(2))**2+abs(a22-(-0.1))**2+abs(a33-(-0.3))**2
    # Non Linear Error is just absolute value of all the coefficent Minus the Linear Error

    #modelerrorNonLinear = sum(sum(abs(model.coefficients()))) - modelerrorLinear
   
    #print('Current Linear error: ' + str(modelerrorLinear))
    #print('Current Non-Linear error: ' + str(modelerrorNonLinear))
   
    return modelerrorLinear, modelerrorNonLinear

def find_min_three_indices(arr):
    if len(arr) <= 3:
        raise ValueError("Input array must have more than three elements")

    # Initialize the minimum three numbers to infinity and their indices to None
    min1, min2, min3 = float('inf'), float('inf'), float('inf')
    idx1, idx2, idx3 = None, None, None

    for idx, num in enumerate(arr):
        if num < min1:
            min3 = min2
            idx3 = idx2
            min2 = min1
            idx2 = idx1
            min1 = num
            idx1 = idx
        elif num < min2:
            min3 = min2
            idx3 = idx2
            min2 = num
            idx2 = idx
        elif num < min3:
            min3 = num
            idx3 = idx

    return idx1, idx2, idx3

itrlimit = 5
totalitr = 0
itr = 0
resolution = 150
SearchWindow = np.linspace(0.0001, 0.9,resolution)
error_stor = [100]
LinearErrors = []
NonLinearErrors = []
plottingaxis =[]

while (itr < itrlimit):
    errors = []
    mins = []
    for eachThreshold in SearchWindow:
        current_model = SINDy_model(eachThreshold,initCondition=[2,0,1])
        error_for_current_model = SINDyThresholdOptimization(current_model)
        errors.append(error_for_current_model[0])
        LinearErrors.append(error_for_current_model[0])
        NonLinearErrors.append(error_for_current_model[1])
        plottingaxis.append(eachThreshold)
        totalitr = totalitr + 1
    error_stor.append(SearchWindow[errors.index(min(errors))])
    mins = find_min_three_indices(errors)
    print("Current min is: " + str(SearchWindow[mins[1]]))
    #if abs(error_stor[-1] - error_stor[-2]) < 0.0000000005:
    #    print("The Itrations have too little change after " + str(itr) + " iterations" )
    #    break
    SearchWindow = np.linspace(SearchWindow[mins[0]],SearchWindow[mins[2]],resolution)
    itr = itr + 1

best_model_threshold = SearchWindow[mins[1]]
current_model = SINDy_model(eachThreshold,initCondition=[2,0,1])
print("Current Linear error is: " + str(SINDyThresholdOptimization(current_model)[0]) )
print("we used Threshold: " + str(SearchWindow[errors.index(min(errors))]))
current_model.print()

print('We collected all the Linear and Non-Linear error, and then we plot!')

plotwindow = 100

xaxis = np.linspace(0,len(LinearErrors[:200]),len(NonLinearErrors[:200]))
fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()

ax1.plot(xaxis,LinearErrors[:200],color="#990000")
# Linear error in red color
ax2.plot(xaxis,NonLinearErrors[:200],color="#009900")
# NonLinear error in blue color

plottingaxisadj = plottingaxis[:200]

ticks = xaxis[::25]
tick_labels = plottingaxisadj[::25]

ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

plt.show()