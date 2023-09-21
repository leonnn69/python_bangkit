import numpy as np

import matplotlib.pyplot as plt

def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))
    
    
    plt.axis([-10, 10, -10, 10])
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0]-0.2+sgn[0], v[1]-0.2+sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

v = np.array([[1],[3]])
# Arguments: list of vectors as NumPy arrays, labels, colors.
plot_vectors([v], [f"$v$"], ["black"])

plot_vectors([v, 2*v, -2*v], [f"$v$", f"$2v$", f"$-2v$"], ["black", "green", "blue"])

# SUM of vector
v = np.array([[1],[3]])
w = np.array([[4],[-1]])

plot_vectors([v, w, v + w], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])

# if u want to use numpy use the code down below.
# plot_vectors([v, w, np.add(v, w)], [f"$v$", f"$w$", f"$v + w$"], ["black", "black", "red"])

# norm v = sqrt(1^2 + 3+2)
print("Norm of a vector v is", np.linalg.norm(v))

# Dot Product using Python
x = [1, -2, -5]
y = [4, 3, -1]

def dot(x, y):
    s = 0
    for xi, yi in zip(x, y):
        s += xi * yi
    return s

print("The dot product of x and y is", dot(x, y))

# quick way to calculate it using function np.dot():
print("np.dot(x,y) function returns dot product of x and y:", np.dot(x, y)) 

# Note that you did not have to define vectors  𝑥  and  𝑦  as NumPy arrays, 
# the function worked even with the lists. But there are alternative functions in Python, 
# such as explicit operator @ for the dot product, which can be applied only to the NumPy arrays. 
# You can run the following cell to check that.
print("This line output is a dot product of x and y: ", np.array(x) @ np.array(y))

print("\nThis line output is an error:")
try:
    print(x @ y)
except TypeError as err:
    print(err)

# Let's redefine vectors  𝑥  and  𝑦  as NumPy arrays to be safe:
x = np.array(x)
y = np.array(y)

# Speed of Calculations in Vectorized Form

# Let's perform a simple experiment to compare their speed. Define new vectors  𝑎  and  𝑏  of the same size  1,000,000 :
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Use time.time() function to evaluate amount of time (in seconds) required to calculate dot product using the function dot(x,y) 
# which you defined above:
import time
tic = time.time()
c = dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the loop version:" + str(1000*(toc-tic)) + " ms")

# Now compare it with the speed of the vectorized versions:
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the vectorized version, np.dot() function: " + str(1000*(toc-tic)) + " ms")

tic = time.time()
c = a @ b
toc = time.time()
print("Dot product: ", c)
print ("Time for the vectorized version, @ function: " + str(1000*(toc-tic)) + " ms")

# You can see that vectorization is extremely beneficial in terms of the speed of calculations!

# contoh dot
a = np.array([1,2,3])
b = np.array([4,5,6])
# 4 + 10 + 18
c = dot(a, b)
print(c)