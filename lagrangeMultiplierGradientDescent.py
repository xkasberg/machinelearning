

import numpy as np
from scipy import optimize

"""This script uses lagrange multipliers to find global extremum """



#Function we are seeking extremum of
def f(x, y):
    return np.exp(-(2*x*x + y*y - x*y) / 2)

#Constraint function
def g(x, y):
    return x*x + 3*(y+1)**2 - 1

#their derivatives
def dfdx (x, y) :
    return 1/2 * (-4*x + y) * f(x, y)

def dfdy (x, y) :
    return 1/2 * (x - 2*y) * f(x, y)

def dgdx (x, y) :
    return 2*x
    
def dgdy (x, y) :
    return 6*(y + 1)


def DL (xyλ) :
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            - g(x, y)
        ])

(x0, y0, λ0) = (-1, -1, 0)
x, y, λ = optimize.root(DL, [x0, y0, λ0]).x
print("x = %g" % x)
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))

