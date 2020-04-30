import numpy as np
import matplotlib.pyplot as plot


"""trains a neural network to draw a curve by implementing backpropagation by the chain rule 
to calculate Jacobians of the cost function.

The neural network will then be trained using a (pre-implemented) stochastic steepest descent method, 
and will draw a series of curves to show the progress of the training.
."""

"""This Curve takes one input variable: The Amount travelled along hte curve from 0 to 1;
    and returns two output variables: the 2D coordinates of the position of points on the curve"""


"""This Networks has 2 hidden Layers with 6 & 7 Neurons respectively"""

"""Function to calculate the Jacobian of the cost function with respect to weights
and Biases"""

#Activation Function
sigma = lambda z : 1 / (1 + np.exp(-z))

#Activation Function derivative
d_sigma = lambda z : np.cosh(z/2)**(-2) / 4


def reset_network(n1 = 6, n2 = 7, random=np.random):
    """Initializes the network with its structure,
    Will reset Training already done"""
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2

def network_function(a0):
    """Feeds forwards the activations from the previous layer. 
    Returns all weighed sums and activations"""
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3

def cost(x, y):
    """
    x and y's are vectors, therefor the sum of squared differences is its modulus
    Returns the average cost function over the training set """
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size



def j_W3(x, y):
    """x and y are vectors; Returns the Jacobian of the third layer Weights
    Where the Jacobian vector is given by the total partial derivative of the Cost Function 
    in respect to the the third Layer Weights:
    or "JW3 = dC/dW3 = (dC/da3) * (da3 / dz3) * (dz3 / dW3)"
    d is the partial derivative
    
    the partial derivatives are given by each expression:
    dC/da3 = 2(a3 - y)
    da3/dz3 = d_sigma(z3)
    dz3/dW3 = a2

    """
    #gets all the activations and weighted sums at each layer of the network.
    print(x)
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    
    #calculates J using the expressions above.
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    #takes the dot product along the axis that holds the training examples with the final partial derivative
    # and divide by the number of training examples, for the average over all training examples.
    J = J @ a2.T / x.size
    return J


def j_W2 (x, y):
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)    
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J

def j_W1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = J @ a0.T / x.size
    return J

def j_b3(x, y):
    """x and y are vectors; Returns the Jacobian of the third layer Weights
    Where the Jacobian vector is given by the total partial derivative of the Cost Function 
    in respect to the the third Layer Weights:
    or "JW3 = dC/dW3 = (dC/da3) * (da3 / dz3) * (dz3 / db3)"
    d is the partial derivative
    
    the partial derivatives are given by each expression:
    dC/da3 = 2(a3 - y)
    da3/dz3 = d_sigma(z3)
    dz3/db3 = 1

    """
    #gets all the activations and weighted sums at each layer of the network.
    print(x)
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    
    #calculates J using the expressions above.
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)

    #Sums over training data and divides by Size of training data to get the average
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def j_b2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

def J_b1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


reset_network()
x= np.array([ 0., 0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.10,   0.11,
   0.12,  0.13,  0.14,  0.15,  0.16,  0.17,  0.18,  0.19,  0.20,  0.21,  0.22,  0.23,
   0.24,  0.25,  0.26,  0.27,  0.28,  0.29,  0.30,  0.31,  0.32,  0.33,  0.34,  0.35,
   0.36,  0.37,  0.38,  0.39,  0.40,  0.41,  0.42,  0.43,  0.44,  0.45,  0.46,  0.47,
   0.48,  0.49,  0.50,  0.51,  0.52,  0.53,  0.54,  0.55,  0.56,  0.57,  0.58,  0.59,
   0.60,  0.61,  0.62,  0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.70,  0.71,
   0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.80,  0.81,  0.82,  0.83,
   0.84,  0.85,  0.86,  0.87,  0.88,  0.89,  0.90,  0.91,  0.92,  0.93,  0.94,  0.95,
   0.96,  0.97,  0.98,  0.99])

y = np.array([[0.50000000, 0.50009902, 0.50078751, 0.50263171, 0.50615226, 0.51180340,
              0.51995466, 0.53087547, 0.54472343, 0.56153657, 0.58122992, 0.60359653,
              0.62831281, 0.65494819, 0.68297861, 0.71180340, 0.74076505, 0.76917106,
              0.79631710, 0.82151087, 0.84409548, 0.86347181, 0.87911897, 0.89061206,
              0.89763674, 0.90000000, 0.89763674, 0.89061206, 0.87911897, 0.86347181,
              0.84409548, 0.82151087, 0.79631710, 0.76917106, 0.74076505, 0.71180340,
              0.68297861, 0.65494819, 0.62831281, 0.60359653, 0.58122992, 0.56153657,
              0.54472343, 0.53087547, 0.51995466, 0.51180340, 0.50615226, 0.50263171,
              0.50078751, 0.50009902, 0.50000000, 0.49990098, 0.49921249, 0.49736829,
              0.49384774, 0.48819660, 0.48004534, 0.46912453, 0.45527657, 0.43846343,
              0.41877008, 0.39640347, 0.37168719, 0.34505181, 0.31702139, 0.28819660,
              0.25923495, 0.23082894, 0.20368290, 0.17848913, 0.15590452, 0.13652819,
              0.12088103, 0.10938794, 0.10236326, 0.10000000, 0.10236326, 0.10938794,
              0.12088103, 0.13652819, 0.15590452, 0.17848913, 0.20368290, 0.23082894,
              0.25923495, 0.28819660, 0.31702139, 0.34505181, 0.37168719, 0.39640347,
              0.41877008, 0.43846343, 0.45527657, 0.46912453, 0.48004534, 0.48819660,
              0.49384774, 0.49736829, 0.49921249, 0.49990098],
              [0.62500000, 0.62701541, 0.63296789, 0.64258068, 0.65540709, 0.67085156,
               0.68819755, 0.70664083, 0.72532628, 0.74338643, 0.75997967, 0.77432624,
               0.78574006, 0.79365515, 0.79764521, 0.79743558, 0.79290745, 0.78409411,
               0.77116996, 0.75443315, 0.73428307, 0.71119423, 0.68568811, 0.65830476,
               0.62957574, 0.60000000, 0.57002377, 0.54002570, 0.51030758, 0.48109110,
               0.45252033, 0.42466948, 0.39755510, 0.37115155, 0.34540857, 0.32026952,
               0.29568895, 0.27164821, 0.24816805, 0.22531726, 0.20321693, 0.18203995,
               0.16200599, 0.14337224, 0.12642077, 0.11144335, 0.09872490, 0.08852676,
               0.08107098, 0.07652676, 0.07500000, 0.07652676, 0.08107098, 0.08852676,
               0.09872490, 0.11144335, 0.12642077, 0.14337224, 0.16200599, 0.18203995,
               0.20321693, 0.22531726, 0.24816805, 0.27164821, 0.29568895, 0.32026952,
               0.34540857, 0.37115155, 0.39755510, 0.42466948, 0.45252033, 0.48109110,
               0.51030758, 0.54002570, 0.57002377, 0.60000000, 0.62957574, 0.65830476,
               0.68568811, 0.71119423, 0.73428307, 0.75443315, 0.77116996, 0.78409411,
               0.79290745, 0.79743558, 0.79764521, 0.79365515, 0.78574006, 0.77432624,
               0.75997967, 0.74338643, 0.72532628, 0.70664083, 0.68819755, 0.67085156,
               0.65540709, 0.64258068, 0.63296789, 0.62701541]])


j_W3(x, y)