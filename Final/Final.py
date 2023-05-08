"""
George Garopoulos
20200017@student.act.edu
Final Exam program for CS 340
"""

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt

#Import the training data
training_data = np.genfromtxt("training_data_set.txt", delimiter=",")

#Import the input data
input_data = np.genfromtxt("input_data_set.txt", delimiter=",")

#Make the neurals
num_input_neurons = 10
num_hidden_neurons = 8
num_output_neurons = 2

#Make a learning Rate
learning_rate = 0.1

#Make it deterministic
np.random.seed(1)

#Initialise the weights and biases
hidden_weights = np.random.randn(num_input_neurons, num_hidden_neurons)
hidden_biases = np.random.randn(1, num_hidden_neurons)
output_weights = np.random.randn(num_hidden_neurons, num_output_neurons)
output_biases = np.random.randn(1, num_output_neurons)

#Make a list for the errors
error_sum = np.empty(1000)

#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Start the loop
for epoch in range(1000):
    #Perform black magic
    hidden_layer_input = np.dot(training_data[:, :-2], hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
    output_layer_output = sigmoid(output_layer_input)

    error = training_data[:, -2:] - output_layer_output

    error_sum[epoch] = np.sum(error)

    #If its the 100th loop print the error
    if epoch % 100 == 0:
        print("epoch:", epoch, "error:", error_sum[epoch])
    #More magic
    d_output = error * sigmoid_derivative(output_layer_output)
    d_hidden = np.dot(d_output, output_weights.T) * sigmoid_derivative(
        hidden_layer_output
    )
    output_weights += learning_rate * np.dot(hidden_layer_output.T, d_output)
    output_biases += learning_rate * np.sum(d_output, axis=0, keepdims=True)
    hidden_weights += learning_rate * np.dot(training_data[:, :-2].T, d_hidden)
    hidden_biases += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

#use the input data and the weights to make an output
l0 = input_data
l1 = sigmoid(np.dot(l0, hidden_weights))
l2 = sigmoid(np.dot(l1, output_weights))

#make the output data array
output_data = np.empty((640, 12))
output_data[:, 0:10] = input_data[:, 0:10]
output_data[:, 10:12] = l2

#Save it to the txt
np.savetxt("output_data.txt", output_data, fmt="%.0f", delimiter=",")

#Print the plot
plt.plot(error_sum)
plt.xlabel("Number of epochs")
plt.ylabel("Error")
plt.show()