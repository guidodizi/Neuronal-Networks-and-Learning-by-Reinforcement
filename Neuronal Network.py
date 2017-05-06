from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, expected_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = []
	for i in range(len(expected_outputs)):
		return_value = expected_outputs[i]
		weights = [random() for i in range(n_hidden + 1)]
		output_layer.append({'weights': weights, 'retrun_value': return_value})
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
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

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
				errors.append(expected[j] - neuron['output'])
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
def train_network(network, train, l_rate, n_epoch, expected_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(len(expected_outputs))]
			expected[expected_outputs.index(row[-1])] = 1
		 	sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row, expected_outputs):
	outputs = forward_propagate(network, row)
	index_output = outputs.index(max(outputs))
	return expected_outputs[index_output]

# Test training backprop algorithm
seed(1)
dataset = createDataset(firstFunction, 40, -1, 1)
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
#number of variables on input
n_inputs = len(dataset[0]) - 1
# #Number of possible output values
# n_outputs = len(set([row[-1] for row in dataset]))
expected_outputs = [row[-1] for row in dataset]
network = initialize_network(n_inputs, 2, expected_outputs)
train_network(network, dataset, 0.5, 100000, expected_outputs)

for row in dataset:
	prediction = predict(network, row, expected_outputs)
	print('Expected=%.5f, Got=%.5f' % (row[-1], prediction))

# for layer in network:
# 	print(layer)
