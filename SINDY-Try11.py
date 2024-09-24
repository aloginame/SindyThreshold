# -*- coding: utf-8 -*-
"""
Created on Tue May 14 07:58:11 2024

@author: Tom
"""

import pysindy as ps
from decimal import *


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
   
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=Threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order)
        )
    # Fit the model
    model.fit(x_train, t=dt)
   
    return model
# line 51 to 61: what this do ? what line 70 "model" differ from line 97 "models=[]"

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

    modelerrorLinear = abs(a11-(-0.1))+abs(a12-(-2))+abs(a21-(2))+abs(a22-(-0.1))+abs(a33-(-0.3))
    # Non Linear Error is just absolute value of all the coefficent Minus the Linear Error
    #
    modelerrorNonLinear = sum(sum(abs(model.coefficients()))) - modelerrorLinear
   
    #print('Current Linear error: ' + str(modelerrorLinear))
    #print('Current Non-Linear error: ' + str(modelerrorNonLinear))
   
    return modelerrorLinear, modelerrorNonLinear
   
# I am keeping this code above for a quick check now we are going to
# apply my Search Algorithm to find the Optimal Threshold in 3 digits
# as the following:
# if suppress the line 97 to 119, your output is different.
ndecimals = 3      # this is not effected to your following code; If set ndecimals = 3 or 5, why the output for the model if different
print(range(ndecimals)) # this just give you [0,5] for "ndecimals =5" ---it is pointless
#initSearchLineSpace = np.array([0.01])  # 10^-2
#initSearchLineSpace = np.append(initSearchLineSpace, np.linspace(0.1,0.9,9))

initSearchLineSpace = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
FixedLineSpace = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# I am finding min linear error
errorsLinear = []
errorsNonLinear = []
plottingaxis = []

inicond = [2,0,1]

for currentdigit in range(ndecimals):  # why set the range in "range(ndecimals)"
    errors = []
    
    # Loop to find the initial array
    for value in initSearchLineSpace:
        plottingaxis.append(value)
        TestThreshold = value
        currentmodel = SINDy_model(TestThreshold, inicond)
       
        errors.append(SINDyThresholdOptimization(currentmodel)[0])
       
        errorsLinear.append(SINDyThresholdOptimization(currentmodel)[0])
        errorsNonLinear.append(SINDyThresholdOptimization(currentmodel)[1])
       
    guess = initSearchLineSpace[errors.index(min(errors))]
    
    tempSpace = []
    """
    tempSpace.append(float(Decimal(str(guess)) - Decimal("0.1")**Decimal(str(int(currentdigit+2))) + Decimal("0.1")**Decimal(str(int(currentdigit+3)))   ))
    for eachvalue in FixedLineSpace[1:]:
        tempSpace.append(float(Decimal(str(guess)) + Decimal(str(eachvalue))*Decimal("0.1")**Decimal(str(currentdigit+1))))
    initSearchLineSpace = tempSpace
    print('Next Search Space is:')
    print(initSearchLineSpace)
   
    """
    if errors.index(min(errors)) == 0:
        print('for the '+ str(int(currentdigit+1)) +'th digit the best is 0 so we go one more digit and current best is: ' + str(guess))
        tempSpace.append(float( Decimal( str(guess) ) - Decimal("0.1")**Decimal(str(int(currentdigit+2))) + Decimal("0.1")**Decimal(str(int(currentdigit+3)))   ))
        for eachvalue in FixedLineSpace[1:]:
            tempSpace.append(float(Decimal(str(guess)) - Decimal("0.1")**Decimal(str(int(currentdigit+2))) + Decimal(str(eachvalue))*Decimal("0.1")**Decimal(str(int(currentdigit+1)))))
        initSearchLineSpace = tempSpace
        if currentdigit < ndecimals-1:
            print('Next Search Space is:')
            print(initSearchLineSpace)
        else:
            print('We reached ' + str(ndecimals) +' decimal places and we will not use the next search space')
        #initSearchLineSpace = initSearchLineSpace * 0.1  # (10^-1)^5 decimals then 10^-2 *10^-5 = 10^-7 so should be 7 digits but 19 digits for 0.0013...03 in output
    else:
        print('for the '+ str(int(currentdigit+1)) +'th digit the best is ' + str(guess))
        tempSpace.append(float(Decimal(str(guess)) + Decimal("0.1")**Decimal(str(int(currentdigit+3)))) )
        for eachvalue in FixedLineSpace[1:]:
            tempSpace.append(float( Decimal(str(guess)) + Decimal(str(eachvalue))*Decimal("0.1")**Decimal(str(int(currentdigit+1)))))
        initSearchLineSpace = tempSpace
        if currentdigit < ndecimals-1:
            print('Next Search Space is:')
            print(initSearchLineSpace)
        else: 
            print('We reached ' + str(ndecimals) +' decimal places and we will not use the next search space')
    
        #initSearchLineSpace = guess + initSearchLineSpace * 0.1
# line 120 to 145 not doing the right thing because the output for 5 decimals places. How come has 0.0012...03 which has 19 digits to it.


"""
initSearchLineSpace = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for currentdigit in range(ndecimals):  # why set the range in "range(ndecimals)"
    errors = []
    # Loop to find the initial array
    for value in initSearchLineSpace:
        TestThreshold = value
        currentmodel = SINDy_model(TestThreshold, [2,0,1])
       
        errors.append(SINDyThresholdOptimization(currentmodel)[0])
       
    guess = initSearchLineSpace[errors.index(max(errors))]
    print('for the '+ str(int(currentdigit)) +'th digit the worst is ' + str(guess))
    tempSpace = []
    if errors.index(min(errors)) == 0:
        for eachvalue in initSearchLineSpace:
            tempSpace.append(float(Decimal(str(guess))+Decimal(str(eachvalue))*Decimal("0.1")))
        initSearchLineSpace = tempSpace # (10^-1)^5 decimals then 10^-2 *10^-5 = 10^-7 so should be 7 digits but 19 digits for 0.0013...03 in output
    else:
        for eachvalue in initSearchLineSpace:
            tempSpace.append(float(Decimal(str(eachvalue))*Decimal("0.1")))
        initSearchLineSpace = tempSpace

"""

# Now, We have found the optimal to (3)ndecimels threshold and now we will print it

print('For initial condition: ' + str(inicond) + ' for ' + str(ndecimals) + ' decimals places')
print('We found optimal threshold as: ' + str(guess))
print('Printing the model: ')

bestmodel = SINDy_model(guess,inicond)
SINDyThresholdOptimization(bestmodel)
bestmodel.print()

print('We collected all the Linear and Non-Linear error, and then we plot!')

xaxis = np.linspace(0,len(errorsLinear),len(errorsLinear))
fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()

ax1.plot(xaxis,errorsLinear,color="#990000")
# Linear error in red color
ax2.plot(xaxis,errorsNonLinear,color="#009900")
# NonLinear error in blue color

ax1.set_xticks(xaxis)
ax1.set_xticklabels(plottingaxis, rotation=45, ha='right')

plt.show()

# the code is not doing the right thing.
# for the output: why we found optimal threshold as: 0.0013010000000000003 has 19 digits instead of 5 decimals places for initial condition: [2, 0, 1].