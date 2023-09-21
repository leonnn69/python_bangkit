import numpy as np

# the question is   -x + 3y = 7
#                   3x - 2y = 1
A = np.array([
    [-1, 3],
    [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A = \n", A,
      "\nArray b = \n", b)

# check dimension
print(f"Shape of A = {A.shape}",
      f"\nShape of b = {b.shape}")

# find values x and y
x = np.linalg.solve(A, b)
print(f"Solution = {x}")

# find the determinant
d = np.linalg.det(A)
print(f"Determinant of matrix A = {d:.2f}")

# if want to use elimination mode
A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)

print(A_system[0])

# implementation of elimination mode

# Function .copy() is used to keep the original matrix without any changes.
A_system_res = A_system.copy()

A_system_res[1] = 3 * A_system_res[0] + A_system_res[1]
print(A_system_res)

A_system_res[1] = 1/11 * A_system_res[1]
print(A_system_res)

# The original matrix A_system was constructed stacking horizontally matrix  𝐴  and array  𝑏 .
# You can read the last row as  0x + 1y = 2 , thus  y = 2 . 
# And the first row as  −1x + 3y = 7 . Substitute  y = 2 , you will get  x = −1 .

# graphical representation of the solution
import matplotlib.pyplot as plt

def plot_lines(M):
    x_1 = np.linspace(-10,10,100)
    x_2_line_1 = (M[0,2] - M[0,0] * x_1) / M[0,1]
    x_2_line_2 = (M[1,2] - M[1,0] * x_1) / M[1,1]
    
    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_1, x_2_line_1, '-', linewidth=2, color='#0075ff',
        label=f'$x_2={-M[0,0]/M[0,1]:.2f}x_1 + {M[0,2]/M[0,1]:.2f}$')
    ax.plot(x_1, x_2_line_2, '-', linewidth=2, color='#ff7300',
        label=f'$x_2={-M[1,0]/M[1,1]:.2f}x_1 + {M[1,2]/M[1,1]:.2f}$')

    A = M[:, 0:-1]
    b = M[:, -1::].flatten()
    d = np.linalg.det(A)

    if d != 0:
        solution = np.linalg.solve(A,b) 
        ax.plot(solution[0], solution[1], '-o', mfc='none', 
            markersize=10, markeredgecolor='#ff0000', markeredgewidth=2)
        ax.text(solution[0]-0.25, solution[1]+0.75, f'$(${solution[0]:.0f}$,{solution[1]:.0f})$', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.xlabel('$x_1$', size=14)
    plt.ylabel('$x_2$', size=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()

plot_lines(A_system)

# System of Linear Equations with No Solutions

# the question =    -x + 3y = 7
#                   3x - 9y = 1

# transform into array
A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

# find the determinant
d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}")

# find if the code singular or no singular
try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)

# try the elimination mode
A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(A_2_system)

# copy() matrix.
A_2_system_res = A_2_system.copy()

# Multiply row 0 by 3 and add it to the row 1.
A_2_system_res[1] = 3 * A_2_system_res[0] + A_2_system_res[1]
print(A_2_system_res)

# The last row will correspond to the equation  0=22  which has no solution.
# Thus the whole linear system  (5)  has no solutions. 

# Let's see what will be on the graph.
plot_lines(A_2_system)