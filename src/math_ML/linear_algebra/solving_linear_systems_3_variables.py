import numpy as np

# The question
#   4x_1 - 3x_2 + x_3 = -10
#   2x_1 + 1x_2 + 3x_3 = 0
#   -x_1 + 2x_2 - 5x_3 = 17

A = np.array([
    [4, -3, 1],
    [2, 1, 3],
    [-1, 2, -5]
], dtype= np.dtype(float))

b = np.array([-10, 0, 17], dtype= np.dtype(float))

print("Matrix A = \n", A,
      "\nArray b = \n", b)

# Check the dimensions of  𝐴  and  𝑏  using shape() function:
print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")

# Now use np.linalg.solve(A, b) function to find the solution of the system  (1)
x = np.linalg.solve(A, b)
print(f"Solution: {x}")

# Let's calculate the determinant using np.linalg.det(A) function:
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")

# Solving System of Linear Equations using Row Reduction

# Preparation for Row Reduction
A_system = np.hstack((A, b.reshape((3,1))))
print(A_system)

# Let's review elementary operations, which do not change the solution set of a linear system:
# - Multiply any row by a non-zero number
# - Add two rows and exchange one of the original rows with the result of the addition
# - Swap rows

# exchange row_num of the matrix M with its multiple by row_num_multiple
def MultiplyRow(M, row_num, row_num_multiple):
    M_new = M.copy()
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new

print("Original matrix:\n", A_system,
      "\nMatrix after its third row is multiplied by 2:\n", MultiplyRow(A_system, 2, 2))

# # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2,
def AddRows(M, row_num_1, row_num_2, row_num_multiple):
    M_new = M.copy()
    M_new[row_num_2] = M_new[row_num_1] * row_num_multiple + M_new[row_num_2]
    return M_new

print("Original matrix:\n", A_system,
      "\nMatrix after exchange of the third row with the sum of itself and second row multiplied by 1/2:\n", 
      AddRows(A_system, 1, 2, 1/2))

# exchange row_num_1 and row_num_2 of the matrix M
def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new

print("Original matrix:\n", A_system,
      "\nMatrix after exchange its first and third rows\n", SwapRows(A_system, 0, 2))

# Row Reduction and Solution of the Linear System

print(A_system)

A_ref = SwapRows(A_system,0,2)
# Note: ref is an abbreviation of the row echelon form (row reduced form)
print(A_ref)

# multiply row 0 of the new matrix A_ref by 2 and add it to the row 1
A_ref = AddRows(A_ref,0,1,2)
print(A_ref)

# multiply row 0 of the new matrix A_ref by 4 and add it to the row 2
A_ref = AddRows(A_ref,0,2,4)
print(A_ref)

# multiply row 1 of the new matrix A_ref by -1 and add it to the row 2
A_ref = AddRows(A_ref,1,2,-1)
print(A_ref)

# multiply row 2 of the new matrix A_ref by -1/12
A_ref = MultiplyRow(A_ref,2,-1/12)
print(A_ref)

print(A_ref[1,1])
# so the answer
x_3 = -2
x_2 = (A_ref[1,3] - A_ref[1,2] * x_3) / A_ref[1,1]
x_1 = (A_ref[0,3] - A_ref[0,2] * x_3 - A_ref[0,1] * x_2) / A_ref[0,0]

print(x_1, x_2, x_3)

# Example of System of Linear Equations with No Solutions
#   1x_1 + 1x_2 + 1x_3 = 2
#          1x_2 - 3x_3 = 1
#   2x_1 + 1x_2 + 5x_3 = 0 

A_2= np.array([
        [1, 1, 1],
        [0, 1, -3],
        [2, 1, 5]
    ], dtype=np.dtype(float))

b_2 = np.array([2, 1, 0], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")

# np.linalg.solve() function will give an error due to singularity.
x_2 = np.linalg.solve(A_2, b_2) 

# lets manual check
A_2_system = np.hstack((A_2, b_2.reshape((3, 1))))
print(A_2_system)

# multiply row 0 by -2 and add it to the row 1
A_2_ref = AddRows(A_2_system,0,2,-2)
print(A_2_ref)

# add row 1 of the new matrix A_2_ref to the row 2
A_2_ref = AddRows(A_2_ref,1,2,1)
print(A_2_ref)