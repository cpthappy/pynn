"""Simple implementation of a neural network
"""
import numpy
import scipy.special


class NeuralNetwork(object):
    """class for representing a neural network

    [description]
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """[summary]

        [description]

        Arguments:
            input_nodes {[type]} -- [description]
            hidden_nodes {[type]} -- [description]
            output_nodes {[type]} -- [description]
            learning_rate {[type]} -- [description]
        """

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.wih = (numpy.random.rand(
            self.hidden_nodes, self.input_nodes) - 0.5)
        self.who = (numpy.random.rand(
            self.output_nodes, self.hidden_nodes) - 0.5)

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """[summary]

        [description]

        Arguments:
            inputs_list {[type]} -- [description]
            targets_list {[type]} -- [description]
        """

        targets = numpy.array(targets_list, ndmin=2).T
        inputs = numpy.array(inputs_list, ndmin=2).T
        final_outputs, hidden_outputs = self._query_internal(inputs)
        output_errors = targets - final_outputs
        hidden_erors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * \
            numpy.dot((output_errors * final_outputs *
                       (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        self.wih += self.learning_rate * \
            numpy.dot((hidden_erors * hidden_outputs *
                       (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def _query_internal(self, inputs):

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs, hidden_outputs

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        return self._query_internal(inputs)[0]


if __name__ == '__main__':
    NETWORK = NeuralNetwork(3, 3, 3, 0.3)
    print NETWORK.query([3.0, 2.1, -3])
