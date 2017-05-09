import math
import sys
from random import seed
from random import random
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *

# Initialize a network
def initialize_network(n_inputs, n_hidden):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]}]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	#hidden layer
	new_inputs = []
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = math.tanh(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
    # only one node in output layer
	return inputs[0]


# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return 1 - math.tanh(output)**2

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			neuron = layer[0]
			errors.append(expected - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch):
	sum_errors_per_iteration = []
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			output = forward_propagate(network, row)
		 	sum_error += (row[-1]-output)**2
			backward_propagate_error(network, row[-1])
			update_weights(network, row, l_rate)
		sum_errors_per_iteration.append(sum_error)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/2))
	return sum_errors_per_iteration

# Make a prediction with a network
def predict(network, row):
	return forward_propagate(network, row)

def createDataset (function, amount, min, max):
	dataset = []
	if function == third_function:
		cases = int(math.ceil(math.sqrt(amount)))
		print cases
		x_values = np.random.uniform(-1, 1, cases)
		y_values = np.random.uniform(-1, 1, cases)
		for x in x_values:
			for y in y_values:
				example = []
				example.append(x)
				example.append(y)
				example.append(function(x,y))
				dataset.append(example)
	else:
		x_values = np.random.uniform(-1, 1, amount)
		for x in x_values:
			example = []
			example.append(x)
			example.append(function(x))
			dataset.append(example)
	return dataset

def first_function (x):
	return (x**3) - (x**2) + 1 
def second_function (x):
	return math.sin(1.5 * math.pi * x) 
def third_function (x, y):
	return 1 - x**2 - y**2

def get_from_dataset (dataset ,position):
	array= []
	for row in dataset:
		array.append(row[position])
	return array

seed(1)
while True:
	function = input("Choose a number of function: \n 1) f(x) = x^3 - x^2 + 1 \n 2) g(x) = sen(1.5 * PI * x) \n 3) h(x,y) = 1 - x^2 - y^2 \n")
	if function == 1:
		function = first_function
		break
	elif function == 2:
		function = second_function
		break		
	elif function == 3:
		function == third_function
		break
	else:
		print "\nSorry, invalid number\n"
		continue
while True:
    number = input("Enter number of neurons on hidden layer: ")
    try:
        n_hidden = int(number)
        if n_hidden < 0: 
            print("Sorry, number of neurons must be a positive integer, try again")
            continue
        break
    except ValueError:
        print("That's not an int!")

while True:
	number = input("Enter a learning train constant: ")
	try:
		l_train = float(number)
		if l_train < 0: 
			print("Sorry, learning train constant must be a positive integer, try again")
			continue
		elif l_train > 1:  
			print("Sorry, learning train constant is recommended to be less than 1, try again")
			continue
		break
	except ValueError:
			print("That's not an int!")
while True:
	number = input("Enter a number of iterations: ")
	try:
		epochs = int(number)
		if epochs < 0: 
			print("Sorry, learning train constant must be a positive integer, try again")
			continue
		break
	except ValueError:
			print("That's not an int!")   

# function = third_function
# n_hidden = 50
# l_train = 0.1
# epochs = 500

dataset = createDataset(function, 40, -1, 1)
#number of variables on input
n_inputs = len(dataset[0]) - 1
#initalize the network
network = initialize_network(n_inputs, n_hidden)
#returns array of errors, index is iteration
sum_errors = train_network(network, dataset, l_train, epochs)
#array of predictions
predictions = []
for row in dataset:
	prediction = predict(network, row)
	predictions.append(prediction)
	print('Expected=%.5f, Got=%.5f' % (row[-1], prediction))

# inputs1 = get_from_dataset(dataset, 0)
# inputs2 = get_from_dataset(dataset, 1)
# expecteds= get_from_dataset(dataset, -1)

# epochs = list(xrange(500))
# error = Scatter(
# 	x=epochs,
# 	y=sum_errors
# )
# original = Scatter3d(
# 	x=inputs1,
# 	y=inputs2,
# 	z=expecteds
# )
# network_prediction = Scatter3d(
# 	x=inputs1,
# 	y=inputs2,
# 	z=predictions
# )
# data = Data([original, network_prediction])
# data = Data([error])

# py.plot(data, filename = 'Error in third_function 500')

