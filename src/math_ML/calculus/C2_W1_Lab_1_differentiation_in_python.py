# Functions in Python

# pangkat
def f(x):
    return x**2

print(f(3))

# kali
def dfdx(x):
    return 2*x

print(dfdx(3))

# apply in array
import numpy as np

x_array = np.array([1, 2, 3])

print("x: \n", x_array)
print("f(x) = x**2: \n", f(x_array))
print("f'(x) = 2x: \n", dfdx(x_array))

import matplotlib.pyplot as plt

# Output of plotting commands is displayed inline within the Jupyter notebook.
# %matplotlib inline

def plot_f1_and_f2(f1, f2=None, x_min=-5, x_max=5, label1="f(x)", label2="f'(x)"):
    x = np.linspace(x_min, x_max,100)

    # Setting the axes at the centre.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, f1(x), 'r', label=label1)
    if not f2 is None:
        # If f2 is an array, it is passed as it is to be plotted as unlinked points.
        # If f2 is a function, f2(x) needs to be passed to plot it.        
        if isinstance(f2, np.ndarray):
            plt.plot(x, f2, 'bo', markersize=3, label=label2,)
        else:
            plt.plot(x, f2(x), 'b', label=label2)
    plt.legend()

    plt.show()
    
plot_f1_and_f2(f, dfdx)

# Symbolic Differentiation

# Introduction to Symbolic Computation with SymPy
import math

math.sqrt(18)

# pake yg atas hasilnya kurang bagus
# makanya pake yg bawa biar lbh bisa diitung
# This format of module import allows to use the sympy functions without sympy. prefix.
from sympy import *

# This is actually sympy.sqrt function, but sympy. prefix is omitted.
sqrt(18)

# Numerical evaluation of the result is available, and you can set number of the digits to show in the approximated output:
N(sqrt(18),8)

# List of symbols.
x, y = symbols('x y')
# Definition of the expression.
expr = 2 * x**2 - x * y
expr    

# Now you can perform various manipulations with this expression: add or subtract some terms, 
# multiply by other expressions etc., just like if you were doing it by hands:

expr_manip = x * (expr + x * y + x**3)
expr_manip

# You can also expand the expression:
expand(expr_manip)

# Or factorise it:
factor(expr_manip)

# buat itung value nya
expr.evalf(subs={x:-1, y:2})

# This can be used to evaluate a function  𝑓(𝑥)=𝑥^2 :
f_symb = x ** 2
f_symb.evalf(subs={x:3})

# multiple array sama f_symb
try:
    f_symb(x_array)
except TypeError as err:
    print(err)

# It is possible to evaluate the symbolic functions for each element of the array, 
# but you need to make a function NumPy-friendly first:
from sympy.utilities.lambdify import lambdify

f_symb_numpy = lambdify(x, f_symb, 'numpy')

print("x: \n", x_array)
print("f(x) = x**2: \n", f_symb_numpy(x_array))

# Symbolic Differentiation with SymPy
# cari dy/dxnya
diff(x**4,x)

dfdx_composed = diff(exp(-2*x) + 3*sin(3*x), x)
dfdx_composed

# Now calculate the derivative of the function f_symb defined in 2.1 and make it NumPy-friendly:
dfdx_symb = diff(f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')
print("x: \n", x_array)
print("f'(x) = 2x: \n", dfdx_symb_numpy(x_array))

dfdx_abs = diff(abs(x),x)
dfdx_abs

#  Introduction to JAX
import jax
from jax import grad, vmap
import jax.numpy as jnp

# Create a new jnp array and check its type.
x_array_jnp = jnp.array([1., 2., 3.])

print("Type of NumPy array:", type(x_array))
print("Type of JAX NumPy array:", type(x_array_jnp))

# The same array can be created just converting previously defined x_array = np.array([1, 2, 3]), 
# although in some cases JAX does not operate with integers, thus the values need to be converted to floats. 
# You will see an example of it below.
x_array_jnp = jnp.array(x_array.astype('float32'))
print("JAX NumPy array:", x_array_jnp)
print("Type of JAX NumPy array:", type(x_array_jnp))

print(x_array_jnp * 2)
print(x_array_jnp[2])

# set value
try:
    x_array_jnp[2] = 4.0
except TypeError as err:
    print(err)

# yang bener
y_array_jnp = x_array_jnp.at[2].set(4.0)
print(y_array_jnp)

# Automatic Differentiation with JAX
# Time to do automatic differentiation with JAX. 
# The following code will calculate the derivative of the previously defined function  𝑓(𝑥)=𝑥2  at the point  𝑥=3 :
print("Function value at x = 3:", f(3.0))
# klo Derivative fnya diturunin dlu dri x^2 jadi 2x
print("Derivative value at x = 3:",grad(f)(3.0))

# Very easy, right? Keep in mind, please, that this cannot be done using integers. The following code will output an error:
try:
    grad(f)(3)
except TypeError as err:
    print(err)

# Try to apply the grad function to an array, calculating the derivative for each of its elements:
try:
    grad(f)(x_array_jnp)
except TypeError as err:
    print(err)

# buat ke array
dfdx_jax_vmap = vmap(grad(f))(x_array_jnp)
print(dfdx_jax_vmap)

plot_f1_and_f2(f, vmap(grad(f)))

# In the following code you can comment/uncomment lines to visualize the common derivatives. 
# All of them are found using JAX automatic differentiation. The results look pretty good!
def g(x):
#     return x**3
#     return 2*x**3 - 3*x**2 + 5
#     return 1/x
#     return jnp.exp(x)
#     return jnp.log(x)
#     return jnp.sin(x)
    return jnp.cos(x)
#     return jnp.abs(x)
#     return jnp.abs(x)+jnp.sin(x)*jnp.cos(x)

plot_f1_and_f2(g, vmap(grad(g)))

# Computational Efficiency of Symbolic, Numerical and Automatic Differentiation
# In sections 2.3 and 3.2 low computational efficiency of symbolic and numerical differentiation was discussed. 
# Now it is time to compare speed of calculations for each of three approaches. 
# Try to find the derivative of the same simple function  𝑓(𝑥)=𝑥^2 multiple times, 
# evaluating it for an array of a larger size, compare the results and time used:
import timeit, time

x_array_large = np.linspace(-5, 5, 1000000)

tic_symb = time.time()
res_symb = lambdify(x, diff(f(x),x),'numpy')(x_array_large)
toc_symb = time.time()
time_symb = 1000 * (toc_symb - tic_symb)  # Time in ms.

tic_numerical = time.time()
res_numerical = np.gradient(f(x_array_large),x_array_large)
toc_numerical = time.time()
time_numerical = 1000 * (toc_numerical - tic_numerical)

tic_jax = time.time()
res_jax = vmap(grad(f))(jnp.array(x_array_large.astype('float32')))
toc_jax = time.time()
time_jax = 1000 * (toc_jax - tic_jax)

print(f"Results\nSymbolic Differentiation:\n{res_symb}\n" + 
      f"Numerical Differentiation:\n{res_numerical}\n" + 
      f"Automatic Differentiation:\n{res_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_symb} ms\n" + 
      f"Numerical Differentiation:\n{time_numerical} ms\n" + 
      f"Automatic Differentiation:\n{time_jax} ms")

# Try to define some polynomial function, which should not be that hard to differentiate, 
# and compare the computation time for its differentiation symbolically and automatically:
def f_polynomial_simple(x):
    return 2*x**3 - 3*x**2 + 5

def f_polynomial(x):
    for i in range(3):
        x = f_polynomial_simple(x)
    return x

tic_polynomial_symb = time.time()
res_polynomial_symb = lambdify(x, diff(f_polynomial(x),x),'numpy')(x_array_large)
toc_polynomial_symb = time.time()
time_polynomial_symb = 1000 * (toc_polynomial_symb - tic_polynomial_symb)

tic_polynomial_jax = time.time()
res_polynomial_jax = vmap(grad(f_polynomial))(jnp.array(x_array_large.astype('float32')))
toc_polynomial_jax = time.time()
time_polynomial_jax = 1000 * (toc_polynomial_jax - tic_polynomial_jax)

print(f"Results\nSymbolic Differentiation:\n{res_polynomial_symb}\n" + 
      f"Automatic Differentiation:\n{res_polynomial_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_polynomial_symb} ms\n" +  
      f"Automatic Differentiation:\n{time_polynomial_jax} ms")