
import numpy as np

#Test if a 4x4 Matrix is Singular
#IE test if an inverse exists
#Before calculating the inverse
#By Converting Matrix to 'Echelon' Form and
#Testing if this failes by leaving zeros 
#that can't be removed on the leading diagnol

#If an inverse does not exist when we transform the matrix, 
#We lose "information" on the transformation, ie Drop Features
#And cannot return to the original Matrix after the transformation

#Our example Matrix


class MatrixIsSingular(Exception): 
    pass
    """A Square Matrix that is not Invertible"""

def isSingular(A):
    """Where A is a 4x4 numpy Matrix 
    B is a copy of A
    returns a Boolean"""
    B = np.array(A, dtype=np.float_)
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

def fixRowZero(A):
    """A is a square 4x4 matrix, 
    All we require is the first element is equal to 1, 
    We divide row by the value of A[0,0]
    If A[0,0] is 0, we cannot do the division
    We test for this, and if it is True, 
    We add the lower rows to the first before the division
    We iterate through the row until we can do the division
    i.e. A[1,0]
    """
    
    #check if zeroth element is equal to zero, add lower row
    
    if A[0, 0] == 0:
        A[0] = A[0] + A[1]
    if A[0, 0] == 0:
        A[0] = A[0] + A[2]
    if A[0,0] == 0:
        A[0] = A[0] + A[3]
    if A[0,0] == 0:
        raise MatrixIsSingular()

    #divide row 0 by its zeroth element
    A[0] = A[0] / A[0,0]
    print('')
    print("After Transformation on Row 0")
    print(A)
    return A

def fixRowOne(A):
    """Sets the sub-diagonal elements to zero, i.e A[1,0]
    We want the diagonal element to equal to one
    We divide the row by the value of A[1,1]
    We need to test if this is zero
    If True, we add a lower and repeat setting the sub-diagonal elements to Zero
    """
    #set sub diagnonal elements to zero
    A[1] = A[1] - A[1,0] * A[0]

    if A[1,1] == 0:
        A[1] = A[1] + A[2]
        print(A[1])
        A[1] = A[1] - A[1,0] * A[0]
        print(A[1])
    if A[1,1] == 0 :
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        raise MatrixIsSingular()
    
    #divide row 1 by its first element 
    A[1] = A[1] / A[1,1]
    
    print('')
    print("After Transformation on Row 1")
    print(A)
    return A

def fixRowTwo(A):
    """Sets the sub-diagonal elements to zero and diaganol elements to one"""

    #sets sub diaganol elements to zero
    A[2] = A[2] - (A[2,0] * A[0]) 
    A[2] = A[2] - (A[2,1] * A[1])

    if A[2,2] == 0:
        A[2] = A[2] + A[3]
        A[2] = A[2] - A[2,0] * A[0]
        A[2] = A[2] - A[2,1] * A[1]
    if A[2,2] == 0:
        raise MatrixIsSingular()
    
    #divide row 2 by its second element 
    A[2] = A[2] / A[2,2]
    print('')
    print('After Transformation on Row 2')
    print(A)
    return A    

def fixRowThree(A):
    """Sets the sub-diagonal elements to zero, i.e A[2,0]
    We want the diagonal element to equal to one
    We divide the row by the value of A[2,2]
    We need to test if this is zero
    If True, we add a lower and repeat setting the sub-diagonal elements to Zero
    """

    #sets sub diagnonal elements to zero
    A[3] = A[3] - A[3,0] * A[0]
    A[3] = A[3] - A[3,1] * A[1]    
    A[3] = A[3] - A[3,2] * A[2]    


    if A[3,3] == 0:
        raise MatrixIsSingular()
    
    #divide row 2 by its second element 
    A[3] = A[3] / A[3,3]
    print('')
    print("After transformation on Row 3")
    print(A)
    return A  



matrix_a = np.array([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 4],
        [0, 0, 5, 5]
    ], dtype=np.float_)

matrix_b = np.array([
        [0, 7, -5, 3],
        [2, 8, 0, 4],
        [3, 12, 0, 5],
        [1, 3, 1, 3]
    ], dtype=np.float_)

#isSingular(matrix_a)
print("Before Transformation on Row 0")
print(matrix_a)
fixRowZero(matrix_a)

print('')
print("Before Transformation on Row 1")
print(matrix_a)
fixRowOne(matrix_a)

print('')
print("Before Transformation on Row 2")
print(matrix_a)
fixRowTwo(matrix_a)

print('')
print("Before Transformation on Row 3")
print(matrix_a)
fixRowThree(matrix_a)


#isSingular(matrix_b)


