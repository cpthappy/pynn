import csv
import os
import pprint
import network

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
        result.append((label, [1.0 * x/max_value +0.01 for x in data]))
    return result


if __name__ == '__main__':
    TRAIN_DATA = read_dataset(os.path.join("data", "mnist_train_100.csv"))
    TRAIN_DATA = scale_dataset(TRAIN_DATA, 255.0)
    pprint.pprint(TRAIN_DATA)