import csv
import numpy
import os
import network


def read_dataset(path):
    dataset = []
    with open(path, "r") as in_file:
        reader = csv.reader(in_file)
        for line in reader:
            converted = map(int, line)
            dataset.append((converted[0], converted[1:]))
    return dataset

def scale_dataset(data_list, max_value):
    result_list = []
    for label, data in data_list:
        result_list.append((label, [1.0 * x/max_value + 0.01 for x in data]))
    return result_list

def create_target(label, output_nodes):
    target_vector = numpy.zeros(output_nodes) + 0.1
    target_vector[label] = 0.99
    return target_vector

def train_model(model, train_data):
    for label, entry in train_data:
        target = create_target(label, 10)
        model.train(entry, target)

def test_model(uut, test_data):
    count = sum([uut.query(entry).argmax() == label for label, entry in test_data])
    return 1.0*count/len(test_data)

if __name__ == '__main__':
    TRAIN_DATA = read_dataset(os.path.join("data", "mnist_train_100.csv"))
    TRAIN_DATA = scale_dataset(TRAIN_DATA, 255.0)
    TEST_DATA = read_dataset(os.path.join("data", "mnist_test_10.csv"))
    TEST_DATA = scale_dataset(TEST_DATA, 255.0)
    NETWORK = network.NeuralNetwork(784, 200, 10, 0.3)
    train_model(NETWORK, TRAIN_DATA)
    print test_model(NETWORK, TEST_DATA)
   