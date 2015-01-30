#!/usr/bin/python

from fann2 import libfann

connection_rate = 1
learning_rate = 0.07
num_input = 64
num_hidden_1 = 20
num_output = 10

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 100

ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_hidden_1, num_output))
#ann.randomize_weights(ann, -1, 1)
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("../../preproc/data/original.data", max_iterations, iterations_between_reports, desired_error)

ann.save("NN.net")

