import numpy as np

# Matrix Multiplication using Python

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])

B = np.array([[2, 2], [5, 7], [4, 4]])

print("Matrix A (3 by 3):\n", A,
      "\nMatrix B (3 by 2):\n", B)

# You can multiply matrices  𝐴  and  𝐵  using NumPy package function np.matmul():
np.matmul(A,B)

# cara klo itung manual =   [[4 * 2 + 5 * 7 + 4 * 9] [2 * 9 + 7 * 9 + 4 * 9]]
#                           [[2 * 9 + 5 * 1 + 4 * 6] [2 * 9 + 7 * 1 + 4 * 6]]
#                           [[2 * 9 + 5 * 2 + 4 * 3] [2 * 9 + 7 * 2 + 4 * 3]]

# we can also use
A @ B

# kalo B, A gabisa karna dimensionn b di cuman 2 x 3
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)

# another example
try:
    B @ A
except ValueError as err:
    print(err)


x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

np.matmul(x,y)

# You can see that there is no error and that the result is actually a dot product  𝑥⋅𝑦 ! 
# So, vector  𝑥  was automatically transposed into the vector  1×3  and matrix multiplication  𝑥𝑇𝑦  was calculated. 
# While this is very convenient, you need to keep in mind such functionality in Python and pay attention to not use it in a wrong way. 
# The following cell will return an error:

try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)