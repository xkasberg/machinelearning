import numpy as np
import numpy.linalg as la



"""
The Gram-Schmidt process is a method for constructing an orthonormal basis of a space that a set of given vectors span. 
It can also be used to determine the dimension of that space, which may be different than the dimension of the vectors 
themselves or the number of vectors provided to span the space

This Code computes a "Gram Schmidt Basis Vector Set" ie, an Orthonormal vector set, and gives the dimension of a span of vectors
"""


"""
The Gram Schmidt Procedure takes a list of vectors and forms an orthonormal basis from this set. 
As a corollary, the procedure allows us to determine the dimension of the space spanned by the basis vectors, 
which is equal to or less than the space which the vectors sit.
"""

#1x10-14 = 
# 0.00000000000001
verySmallNumber = 1e-14



def gsBasis(A):
    """Performs Gram Schmidt Procedure on a Matrix

    List of vectors is the Columns of A, 
    Iterates through the list and sets the current element to be orthogonal to all
    previous elements before finally Normalizing, ie: setting unit length to 1
    
    Returns a Matrix that is an Orthonormal Basis Vector Set to the original
    """

    # Make B as a copy of A, since we're going to alter it's values.
    B = np.array(A, dtype=np.float_) 


    for i in range(B.shape[1]):

        for j in range(i) :

            #For the ith column, we need to subtract any overlap with our new jth vector.
            #"@" is Python's Matrix Multiplication Operator
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        #Normalizes ith column by dividing it by its modulus, or norm 
        #if there is anything left in column i after the subtraction
        #and therefor, linearly independent of all previous vectors
        #Else, we set the vector to Zero
        if la.norm(B[:, i]) > verySmallNumber :
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])  
            
        
    print("gsA:")
    print(B)
    return B


def dimensions(A):
    """This function uses the Gram Schmidt Procedure to calculate the dimension
    spanned by a list of vectors.
    Since each vector is either zero or normalized to one,
    the sum of all the norms will be the dimension

    Returns the number of dimensions of the Matrix
    """

    return np.sum(la.norm(gsBasis(A), axis=0))

V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)


gsBasis(V)

#Non Square Matrix Test
A = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12,2,1]], dtype=np.float_)


gsBasis(A)
dimensions(A)

B = np.array([[6,2,1,7,5],
              [2,8,5,-4,1],
              [1,-6,3,2,8]], dtype=np.float_)
gsBasis(B)
dimensions(B)


#Linear Dependent Matrix Test
C = np.array([[1,0,2],
              [0,1,-3],
              [1,0,2]], dtype=np.float_)
gsBasis(C)