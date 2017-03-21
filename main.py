import numpy as np
from NearestNeighbor import *

#read first 800 samples (first 2 persons) to a matrix.
raw_data = np.loadtxt(open("DSL-StrongPasswordData.csv", "rb"), dtype='string',  delimiter=",", skiprows=1)
data = raw_data[:800,]

#scramble rows, then split inte training and validation sets.
np.random.shuffle(data)
train_data = data[:600,]
valid_data = data[600:800,]

#Format training labels and data for mathemagic
train_labels = train_data[:,:1]
train_values = train_data[:,4:].astype(float)

#Format validation labels and data for mathemagic
valid_labels = valid_data[:,:1]
valid_values = valid_data[:,4:].astype(float)

print train_labels
print train_values
