import numpy as np

#   2a -  b +  c +  d = 6
#    a + 2b -  c -  d = 3
#   -a + 2b + 2c + 2d = 14
#    a -  b + 2c +  d = 8

#  [ 1a  2b -1c -1d  3.]
#  [ 0.  1b  4c  3d 22.]
#  [ 0.  0.  1c  3d  7.]
#  [-0. -0. -0.  1d  1.]]
# d = 1
# 1c + 3 *1 = 7


A = np.array([
    [2, -1, 1, 1],
    [1, 2, -1, -1],
    [-1, 2, 2, 2],
    [1, -1, 2, 1]
], dtype = np.dtype(float))

b = np.array([
    [6, 3, 14, 8]
], dtype = np.dtype(float))

print(A)
print(b)

c = b.reshape((4, 1))
print(c)
# Find the determinant
d = np.linalg.det(A)
print(f"determinannya = {d:.2f}")

# solve the problem
x = np.linalg.solve(A, c)
print(f"Nilai a, b, c, d= \n{x}")

# stack the matrix
A_system = np.hstack((A, b.reshape((4, 1))))
print(A_system)

def MultiplyRow(M, row_num, row_num_multiple):
    M_new = M.copy()

    M_new[row_num] = M_new[row_num] * row_num_multiple

    return M_new

MultiplyRow(A_system, 0, 2)

def AddRows(M, row_num_1, row_num_2, row_num_multiple):
    M_new = M.copy()

    M_new[row_num_2] = M_new[row_num_1] * row_num_multiple + M_new[row_num_2]

    return M_new

AddRows(A_system, 2, 0, 4)

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()

    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]

    return M_new

SwapRows(A_system, 0, 1)

def augmented_to_ref(A, b):    
    ### START CODE HERE ###
    # stack horizontally matrix A and vector b, which needs to be reshaped as a vector (4, 1)
    A_system = np.hstack((A, b.reshape((4, 1))))
    
    # swap row 0 and row 1 of matrix A_system (remember that indexing in NumPy array starts from 0)
    A_ref = SwapRows(A_system, 0, 1)
    
    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_ref = AddRows(A_ref, 0, 1, -2)
    
    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_ref = AddRows(A_ref, 0, 2,1)
    
    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_ref = AddRows(A_ref, 0, 3, -1)
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 2, 3, 1)

    # swap row 1 and 3 of the new matrix A_ref
    A_ref = SwapRows(A_ref, 1, 3)
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 2, 3, 1)
    
    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_ref = AddRows(A_ref, 1, 2, -4)
    
    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 1, 3, 3)
    
    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_ref = AddRows(A_ref, 3, 2, 2)
    
    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_ref = AddRows(A_ref, 2, 3, -8)
    
    # multiply row 3 of the new matrix A_ref by -1/17
    A_ref = MultiplyRow(A_ref, 3, -1/17)
    ### END CODE HERE ###
    
    return A_ref

A_ref = augmented_to_ref(A, b)

print(A_ref)