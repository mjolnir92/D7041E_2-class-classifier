import numpy as np

class NearestNeighbor:
	def train(self, values, labels, k):
		self.values = values
		self.labels = labels
		self.k = k

	def predict(self, values):
		test = values.shape[0]
		predict = np.zeros(test, dtype = self.labels.dtype)
		for i in xrange(test):
			dist = np.sqrt(np.sum(np.square(self.values - values[i,:]), axis = 1))
			predict[i] = self.kNN(dist, self.k)
		return predict

	def kNN(self, dist, k):
		near = np.argsort(dist)[:k]
		ans = np.zeros(2)
		for n in near:
			ans[self.labels[n]] += 1
		return np.argmax(ans)

	def crossFoldValid(self, values, labels, k_max):
		x = np.split(values, 3)
		y = np.split(labels, 3)
	
		train_val = None
		valid_val = None
		train_lbl = None
		valid_lbl = None

		fold_results = []

		for i in xrange(3):
			print "fold ", i+1
			train_val = np.concatenate((x[i%3], x[(i+1)%3]), axis = 0)
			valid_val = x[(i+2)%3]
			train_lbl = np.concatenate((y[i%3], y[(i+1)%3]), axis = 0)
			valid_lbl = y[(i+2)%3]

			accs = []

			for j in range(1, k_max):
				self.train(train_val, train_lbl, j)
				pred = self.predict(valid_val)
				acc = np.mean(pred == valid_lbl)
				print "acc for k=", j, ": ", acc
				accs.append(acc)

			fold_results.append(accs)

		totals = np.zeros(k_max)

		for k in range(1, k_max):
			totals[k] = fold_results[0][k-1] + fold_results[1][k-1] + fold_results[2][k-1]

		return np.argmax(totals)

