import numpy as np



def a1(a0):
    """1 Node, Neural Network 'NOT' Function 
    a0 must be a 1 or a 0, 
    returns a 0 if input is FALSE, 1 if input is TRUE""" 
    #First we set the state of the network
    sigma = np.tanh
    w1 = -5
    b1 = 5
    #Then we return the neuron activation.
    return sigma(w1 * a0 + b1)
  
x=1
a1(x)


def oneLayerNN():
    """Extends Notation into Matrix & Vector Form"""

    # First set up the network.
    sigma = np.tanh
    W = np.array([[-2, 4, -1],[6, 0, -3]])
    b = np.array([0.1, -2.5])

    # Define our input vector
    a0 = np.array([0.3, 0.4, 0.1])
    a1 = sigma(W@a0 +b)
    print(a1)

    return a1

oneLayerNN()
# Calculate the values by hand,
# and replace a1_0 and a1_1 here (to 2 decimal places)
# (Or if you feel adventurous, find the values with code!)
#a1 = np.array([a1_0, a1_1])