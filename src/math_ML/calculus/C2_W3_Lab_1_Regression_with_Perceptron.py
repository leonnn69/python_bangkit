# Regression with Perceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(3)

# Dataset
path = 'assets/tvmarketing.csv'
adv = pd.read_csv(path)

print(adv)

adv.head()

adv.plot(x='TV', y='Sales', kind='scatter', c='black')
plt.show()

# Column-wise normalization of the dataset can be done for all of the fields at once and is implemented in the following code:
adv_norm = (adv - np.mean(adv))/np.std(adv)

# Plotting the data, you can see that it looks similar after normalization, but the values on the axes have changed:
adv_norm.plot(x='TV', y='Sales', kind="scatter", c='black')
plt.show()


import numpy as np

data = [1, 2, 3, 4, 5]
std_dev = np.std(data)

print("Standard Deviation:", std_dev)

# Save the fields into variables X_norm and Y_norm and reshape them to row vectors:
X_norm = adv_norm['TV']
Y_norm = adv_norm['Sales']

X_norm = np.array(X_norm).reshape((1, len(X_norm)))
Y_norm = np.array(Y_norm).reshape((1, len(Y_norm)))

print(f"The shape of norm X : {np.shape(X_norm)}\n"
      f"The shape of norm Y : {np.shape(Y_norm)}\n"
      f"So, i have {X_norm.shape[1]} training example")

# 2 - Implementation of the Neural Network Model for Linear Regression
# Define two variables:

# n_x: the size of the input layer
# n_y: the size of the output layer
# using shapes of arrays X and Y.

def layer_sizes(x, y):
    n_x = x.shape[0]
    n_y = y.shape[0]
    return(n_x, n_y)

(n_x, n_y)= layer_sizes(X_norm, Y_norm)
print(f"The size of input layer is: n_x = {str(n_x)}\n"
      f"The size of input layer is: n_y = {str(n_y)}")

# 2.2 - Initialize the Model's Parameters
def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_x, n_y)
    b = np.zeros((n_y, 1))

    parameters = {"W" : W,
                  "b" : b}
    
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))


# 2.3 - The Loop
# Implement forward_propagation() following the equation (3) in the section 1.2:
# 𝑍 = 𝑤𝑋+𝑏
# 𝑌̂ = 𝑍,
# forward_propagation

def forward_propagation(X, parameters):
    W = parameters["W"]
    b = parameters["b"]

    Z = np.matmul(W, X) + b
    Y_hat = Z

    return Y_hat

Y_hat = forward_propagation(X_norm, parameters)
print("Some element of output vector = ", Y_hat[0, 0:7])
# Define a cost function $(4)$ which will be used to train the model:
# $$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y_hat.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

print("cost = " + str(compute_cost(Y_hat, Y_norm)))

# Calculate partial derivatives
# \begin{align}
# \frac{\partial \mathcal{L} }{ \partial w } &= 
# \frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)},\\
# \frac{\partial \mathcal{L} }{ \partial b } &= 
# \frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
# \end{align}

def backward_propagation(Y_hat, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    Y_hat -- the output of the neural network of shape (n_y, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(Y_hat, X_norm, Y_norm)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))

# Update parameters
# \begin{align}
# w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\\
# b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b }.
# \end{align}

def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    learning_rate -- learning rate parameter for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))

# 2.4 - Integrate parts 2.1, 2.2 and 2.3 in nn_model() and make predictions
# Build your neural network model in nn_model()
def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # Loop
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        
        # Backpropagation. Inputs: "Y_hat, X, Y". Outputs: "grads".
        grads = backward_propagation(Y_hat, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters_simple = nn_model(X_norm, Y_norm, num_iterations=30, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters_simple["W"]))
print("b = " + str(parameters_simple["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]

# The final model parameters can be used for making predictions, but don't forget about normalization and denormalization.
def predict(X, Y, parameters, X_pred):
    
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Use the same mean and standard deviation of the original training array X.
    if isinstance(X, pd.Series):
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_pred_norm = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(np.mean(X)).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)
    # Make predictions.
    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    # Use the same mean and standard deviation of the original training array Y.
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)
    
    return Y_pred[0]

X_pred = np.array([50, 120, 280])
Y_pred = predict(adv["TV"], adv["Sales"], parameters_simple, X_pred)
print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales:\n{Y_pred}")

# Let's plot the linear regression line and some predictions. The regression line is red and the predicted points are blue.
fig, ax = plt.subplots()
plt.scatter(adv["TV"], adv["Sales"], color="black")

plt.xlabel("$x$")
plt.ylabel("$y$")
    
X_line = np.arange(np.min(adv["TV"]),np.max(adv["TV"])*1.1, 0.1)
Y_line = predict(adv["TV"], adv["Sales"], parameters_simple, X_line)
ax.plot(X_line, Y_line, "r")
ax.plot(X_pred, Y_pred, "bo")
plt.plot()
plt.show()

# 3 - Multiple Linear Regression
df = pd.read_csv('assets/house_prices_train.csv')

# Select the required fields and save them in the variables X_multi, Y_multi:
X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

# Preview the data:
# display(X_multi)
# display(Y_multi)

# Normalize the data:
X_multi_norm = (X_multi - np.mean(X_multi))/np.std(X_multi)
Y_multi_norm = (Y_multi - np.mean(Y_multi))/np.std(Y_multi)

# Convert results to the NumPy arrays, transpose X_multi_norm to get an array of a shape ( 2×𝑚 )
# and reshape Y_multi_norm to bring it to the shape ( 1×𝑚 ):
X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))

# Performance of the Neural Network Model for Multiple Linear Regression
parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iterations=100, print_cost=True)

print("W = " + str(parameters_multi["W"]))
print("b = " + str(parameters_multi["b"]))

W_multi = parameters_multi["W"]
b_multi = parameters_multi["b"]

# Now you are ready to make predictions:
X_pred_multi = np.array([[1710, 7], [1200, 6], [2200, 8]]).T
Y_pred_multi = predict(X_multi, Y_multi, parameters_multi, X_pred_multi)

print(f"Ground living area, square feet:\n{X_pred_multi[0]}")
print(f"Rates of the overall quality of material and finish, 1-10:\n{X_pred_multi[1]}")
print(f"Predictions of sales price, $:\n{np.round(Y_pred_multi)}")