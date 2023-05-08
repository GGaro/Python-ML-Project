"""
George Garopoulos 
20200017
Scrap 1 of Part B for CS 340 project
"""

import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# number of training iterations
training_iterations = 10000

# training loop
for iteration in range(training_iterations):
    # forward propagation
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)

# test the neural network with a new situation.
print('Considering new situation [1, 0, 0] -> ?: ')
print(sigmoid(np.dot(np.array([1, 0, 0]), synaptic_weights)))

# plot the cost
plt.plot(np.squeeze(error))
plt.ylabel('error')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(0.1))
plt.show()

# plot the cost
plt.plot(np.squeeze(outputs))
plt.ylabel('outputs')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(0.1))
plt.show()

# plot the cost
plt.plot(np.squeeze(adjustments))
plt.ylabel('adjustments')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(0.1))
plt.show()