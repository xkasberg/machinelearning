import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)


def pageRank(linkMatrix, d):
    """
    This function ranks pages based off of googles PageRank Algorithm,
    The algorithm is designed to calculate the probability of a crawler landing on a site 
    if it were to click through random links of connected websites

    In this way, the amount of crawlers on a site can be represented as a vector, 
    the amount of crawlers at time t+1 can be represented as the vector being transformed by Transformation M 
    or r @ M 

    Where M is a function of d and the number of sites in the network 

    linkMatrix is an nxn Matrix representing a network of linked sites    
    Its Columns describe probability of getting to a site from itself i.e. Lij > Lj
    These probabilities must sum to 1
    Rows represent probability of getting to a site from anyother, and do not necassarily sum to 1

    d is a float, and is the damping factor; an estimator representing the probability of a 
    crawler entering in a random URL and navigating directly to the site, rather than clicking through

    returns the Matrices principal Eigenvector, or the rank of sites based on probability of navigation to it
    """
    n = linkMatrix.shape[0]
    print(n)

    #sets up a vector of "Crawlers" crawling websites & following links, distributes them equally on each site
    r = 100 * np.ones(n) / n 
    lastR = r
    #J is an n by n matrix of ones
    J = np.ones((n,n))
    #M is the Transform Matrix
    M = d * linkMatrix + ((1-d)/n) * J
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")

    print(r)
    return r

L = np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])


pageRank(L, .5)