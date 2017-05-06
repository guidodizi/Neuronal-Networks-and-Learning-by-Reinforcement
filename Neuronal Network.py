import math
from random import seed
from random import random

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

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			# neuron['output'] = transfer(activation)
			neuron['output'] = math.tanh(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs[0]

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

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
			for j in range(len(layer)):
				neuron = layer[j]
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


def createDataset (function, amount, min, max):
	diff = float (max - min) / amount
	dataset = []
	current = min
	for i in range(amount):
		example = []
		example.append(current)
		example.append(function(current))
		dataset.append(example)
		current += diff
	return dataset

def firstFunction (x):
	return (x**3) - (x**2) + 1 

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			output = forward_propagate(network, row)
		 	sum_error += (row[-1]-output)**2
			backward_propagate_error(network, row[-1])
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error/2))

# Make a prediction with a network
def predict(network, row):
	return forward_propagate(network, row)


# Test training backprop algorithm
seed(1)
dataset = createDataset(firstFunction, 40, -1, 1)

#number of variables on input
n_inputs = len(dataset[0]) - 1

network = initialize_network(n_inputs, 2)
#l_train = 0.5  epochs = 100  => error = 0.150 => ಠ_ಠ
train_network(network, dataset, 0.5, 100)

for row in dataset:
	prediction = predict(network, row)
	print('Expected=%.5f, Got=%.5f' % (row[-1], prediction))

# for layer in network:
# 	print(layer)
