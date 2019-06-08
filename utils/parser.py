import csv
import random
import math
import numpy as np

# returns training and testing datasets
def parseCSV(filename, num_qubits = 8):
	raw = []
	buckets = 2**num_qubits

	window_size = num_qubits // 2
	with open(filename) as csv_data:
		data = csv.reader(csv_data)
		next(data)
		for rows in data:
			raw.append(float(rows[5]))
	# print(math.ceil(math.log(max(raw))/math.log(2)))
	max_val = math.pow(2, math.ceil(math.log(max(raw))/math.log(2)))
	# print(max_val)
	raw.reverse()
	# window_size = 5
	# print(len(raw))
	bucket_size = int(max_val / buckets)
	# print(bucket_size)
	data_pairs = []
	# print(raw)
	raw_np = np.array(raw) // bucket_size
	raw = list(raw_np)
	for i in range(len(raw) - window_size):
		data_pairs.append((raw[i:i+window_size], raw[i + window_size]))

	#random.shuffle(data_pairs)

	# print(data_pairs)

	training_size = len(raw)
	testing_size = len(raw) - training_size

	training = data_pairs[:training_size].copy()
	# testing = data_pairs[training_size:].copy()
	return training

def main():
	training, testing = parseCSV("data/daily_adjusted_FB.csv")

if __name__ == '__main__':
	main()
