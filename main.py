import numpy as np
from NearestNeighbor import *

#read first 800 samples (first 2 persons) to a matrix
raw_data = np.loadtxt(open("DSL-StrongPasswordData.csv", "rb"), dtype='string',  delimiter=",", skiprows=1)
data = raw_data[:800,]

#scramble rows, then split inte training and validation sets
np.random.shuffle(data)
train_data = data[:600,]
valid_data = data[600:800,]

#format training labels and data for mathemagic
train_labels = train_data[:,:1].flatten()
for i in xrange(train_labels.size):
	if train_labels[i] == 's002':
		train_labels[i] = int(0)
	else:
		train_labels[i] = int(1)
train_labels = train_labels.astype(int)
train_values = train_data[:,4:].astype(float)

#format validation labels and data for mathemagic
valid_labels = valid_data[:,:1].flatten()
for i in xrange(valid_labels.size):
	if valid_labels[i] == 's002':
		valid_labels[i] = int(0)
	else:
		valid_labels[i] = int(1)
valid_labels = valid_labels.astype(int)
valid_values = valid_data[:,4:].astype(float)

#instance of class
nn = NearestNeighbor()

#cross fold validate to find best k, then train nn with k, then predict labels
k = nn.crossFoldValid(train_values, train_labels, 101)
print "best k foud to be: ", k
nn.train(train_values, train_labels, k)
pred = nn.predict(valid_values)

#print accuracy
print 'accuracy: %f' % (np.mean(pred == valid_labels))
