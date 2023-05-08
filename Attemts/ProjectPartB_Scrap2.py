"""
George Garopoulos 
20200017
Scrap 2 of Part B for CS 340 project
"""

import numpy as np

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#derivative of sigmoid function
def sigmoid_derivative(x):
    return x*(1-x)

#input data
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

#output data
training_outputs = np.array([[0,1,1,0]]).T

#seed random numbers to make calculation deterministic
np.random.seed(1)

#initialize weights randomly with mean 0
synaptic_weights = 2*np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

#training step
for iteration in range(10000):
    #forward propagation
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    #calculate error
    error = training_outputs - outputs

    #multiply error by input and gradient of the sigmoid function
    #less confident weights are adjusted more through the nature of the function
    adjustments = error * sigmoid_derivative(outputs)

    #update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)

#test
new_inputs = np.array([1,1,0])
output = sigmoid(np.dot(new_inputs, synaptic_weights))

print('New input data: ')
print(new_inputs)
print('Output data: ')
print(output)