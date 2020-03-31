from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    """Calculatese best fit slope,
    xs is numpy array,
    yx is numpy array"""
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    print("M = " + str(m)+", B = " +(str(b)))

    return m, b 

def squared_error(observations, statistics):
    return sum((observations-statistics)**2)

def coefficient_of_determination(observations, statistics):
    y_mean_line = [mean(observations) for i in observations]
    squared_error_y_hat = squared_error(observations, statistics)
    squared_error_y_mean = squared_error(observations, y_mean_line)
    print("R^2: "+str(1 - (squared_error_y_hat/squared_error_y_mean)))
    return 1 - (squared_error_y_hat/squared_error_y_mean)

def r_squared(ys, regression_line):
    """ys and regression_line is a list, 
       returns r_sqaured"""
    se_y_hat = []
    for yi, y_hat, in zip(ys, regression_line):
       term = yi - y_hat
       term_squared = term*term
       se_y_hat.append(term_squared)

    se_y_mean = mean(ys)
    se_y = []
    for y in ys:
        term = se_y_mean - y
        term_squared = term*term
        se_y.append(term_squared)

    sum_se_y_hat = sum(se_y_hat)
    sum_se_y = sum(se_y)
    return 1 - (sum_se_y_hat / sum_se_y)

def create_dataset(how_many, variance, step=2, correlation=False):
    """correlation must be string 'pos', or 'neg'"""
    val = 1
    ys =[]
    for i in range(how_many):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation =='neg':
            val -= step
    
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 80, 2, correlation=True)

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x +b) for x in xs]

r_squared_trial = r_squared(ys, regression_line)

coefficient_of_determination(ys, regression_line)
#print(coefficient_of_determination(ys, regression_line))

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()

#predict_x = 8
#predict_y = (m*predict_x) + b
