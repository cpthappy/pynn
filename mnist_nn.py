import csv
import os
import pprint
import network
import numpy

def read_dataset(path):
    dataset = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            converted = map(int, line)
            dataset.append((converted[0], converted[1:]))
    return dataset

def scale_dataset(data_list, max_value):
    result = []
    for label, data in data_list:
        result.append((label, [1.0 * x/max_value + 0.01 for x in data]))
    return result

def create_target(label, output_nodes):
    target = numpy.zeros(output_nodes) + 0.1
    target[label] = 0.99
    return target

if __name__ == '__main__':
    TRAIN_DATA = read_dataset(os.path.join("data", "mnist_train_100.csv"))
    TRAIN_DATA = scale_dataset(TRAIN_DATA, 255.0)
    TEST_DATA = read_dataset(os.path.join("data", "mnist_test_10.csv"))
    TEST_DATA = scale_dataset(TEST_DATA, 255.0)
    NETWORK = network.NeuralNetwork(784, 200, 10, 0.3)
    
    for label, entry in TRAIN_DATA:
        target = create_target(label, 10)
        NETWORK.train(entry, target)

    for label, entry in TEST_DATA:
        result = NETWORK.query(entry)
        print label, result.argmax()