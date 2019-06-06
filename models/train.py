from models.qubits5 import *
from utils.parser import *

# print(gen_ansatz(0, 0))
def main():
    print(NUM_FEATURES)
    training = parseCSV("data/daily_adjusted_FB.csv")

if __name__ == '__main__':
    main()
