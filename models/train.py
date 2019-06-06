from models.qubits5 import *
from utils.parser import *

# print(gen_ansatz(0, 0))
def main():
    print(NUM_FEATURES)
    training = parseCSV("data/daily_adjusted_FB.csv")
    num_epochs = 50

    for i in range(num_epochs);
        for input in training:
if __name__ == '__main__':
    main()
