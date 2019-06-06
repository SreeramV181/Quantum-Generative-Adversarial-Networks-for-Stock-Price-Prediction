import csv
import random
raw = []
with open('daily_adjusted_MSFT.csv') as csv_data:
	data = csv.reader(csv_data)
	next(data)
	for rows in data:
		raw.append(float(rows[5]))

raw.reverse()
window_size = 5
print(len(raw))
data_pairs = []
for i in range(len(raw) - window_size):
	data_pairs.append((raw[i:i+window_size], raw[i + window_size]))

random.shuffle(data_pairs)

# print(data_pairs)

training_size = int(0.8 * len(raw))
testing_size = len(raw) - training_size

training = data_pairs[:training_size].copy()
testing = data_pairs[training_size:].copy()

print(len(training))
print(len(testing))
print(len(training) + len(testing))