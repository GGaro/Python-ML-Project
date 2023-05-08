"""
George Garopoulos 
20200017
Part B for CS 340 project
"""

import numpy as np
import matplotlib.pyplot as plt

# Global variables

# Network topology
input_layer_size = 0
hidden_layer_size = 0
output_layer_size = 0

# Training data
training_data = []
training_output = []

# Training parameters
learning_rate = 0.1
training_epochs = 0

# Weights
weights_input_hidden = []
weights_hidden_output = []

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Network functions
def init_network():
    global weights_input_hidden
    global weights_hidden_output
    weights_input_hidden = (
        2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
    )
    weights_hidden_output = (
        2 * np.random.random((hidden_layer_size, output_layer_size)) - 1
    )


def forward_propagation(input_data):
    global weights_input_hidden
    global weights_hidden_output
    input_data = float(input_data)
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    return output_layer_output


def back_propagation(input_data, output_data):
    global weights_input_hidden
    global weights_hidden_output
    input_data = float(input_data)
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    output_data = np.float_(output_data)
    output_layer_error = output_data - output_layer_output
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)
    hidden_layer_error = output_layer_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    weights_hidden_output += (
        hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    )
    weights_input_hidden += np.dot(input_data, hidden_layer_delta) * learning_rate


def train_network():
    global training_data
    global training_output
    global training_epochs
    global learning_rate
    global weights_input_hidden
    global weights_hidden_output
    training_progress = []
    init_network()
    for i in range(training_epochs):
        for j in range(len(training_data)):
            back_propagation(training_data[j], training_output[j])
    for i in range(training_epochs):
        for j in range(len(training_data)):
            back_propagation(training_data[j], training_output[j])
        training_progress.append(training_output)
    with open("training_progress.txt", "w") as f:
        for i in range(len(training_progress)):
            f.write("Epoch: " + str(i) + "\n " + str(training_progress[i]) + "\n")


def classify_test_data2():
    global training_data
    global training_output
    global weights_input_hidden
    global weights_hidden_output
    init_network()
    for i in range(len(training_data)):
        output = forward_propagation(training_data[i])
        training_output.insert(i, output)


def display_training_result_graphicsplt():
    global training_data
    global training_output
    global weights_input_hidden
    global weights_hidden_output
    init_network()
    training_progress = []
    for i in range(training_epochs):
        for j in range(len(training_data)):
            back_propagation(training_data[j], training_output[j])
        training_progress.append(training_output)
    training_progress = np.array(training_progress, float)
    plt.plot(training_progress)
    plt.show()


# Menu functions
def enter_network_topology():
    global input_layer_size
    global hidden_layer_size
    global output_layer_size
    input_layer_size = int(input("Enter input layer size: (Default: 10)") or 10)
    hidden_layer_size = int(input("Enter hidden layer size: (Default: 5)") or 5)
    output_layer_size = int(input("Enter output layer size: (Default: 2)") or 2)


def initiate_training_pass():
    global training_data
    global training_output
    global training_epochs
    global learning_rate
    global weights_input_hidden
    global weights_hidden_output
    training_data = []
    training_output = []
    training_epochs = int(
        input("Enter number of training epochs: (Default: 100)") or 100
    )
    learning_rate = float(input("Enter learning rate: (Default: 0.1)") or 0.1)
    with open("training_data.txt", "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")
            training_data.append(line[0])
            training_output.append(line[1:])
    train_network()


def classify_test_data():
    global training_data
    global training_output
    global weights_input_hidden
    global weights_hidden_output
    training_data = []
    training_output = []
    with open("input_data.txt", "r") as f:
        for line in f:
            line = line.strip()
            training_data.append(line)
    classify_test_data2()
    with open("training_output.txt", "w") as f:
        for i in range(len(training_data)):
            f.write(training_data[i] + "," + str(training_output[i]) + "\n")


def display_training_result_graphics():
    global training_data
    global training_output
    global weights_input_hidden
    global weights_hidden_output
    training_data = []
    training_output = []
    with open("training_data.txt", "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")
            training_data.append(line[0])
            training_output.append(line[1:])
    display_training_result_graphicsplt()


# Main function
def main():
    while True:
        print("1. Enter network topology")
        print("2. Initiate a training pass")
        print("3. Classify test data")
        print("4. Display training result graphics")
        print("5. Exit the program")
        try:
            choice = int(input("Enter choice: "))
        except:
            choice = 9
        if choice == 1:
            enter_network_topology()
        elif choice == 2:
            initiate_training_pass()
        elif choice == 3:
            classify_test_data()
        elif choice == 4:
            display_training_result_graphics()
        elif choice == 5:
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()