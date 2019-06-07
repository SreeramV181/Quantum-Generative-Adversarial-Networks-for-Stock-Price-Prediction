from models.qubits5 import *
from utils.parser import *
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *



# print(gen_ansatz(0, 0))
def main():
    #Constants
    NUM_EPOCHS = 50
    EPS = 1e-2

    training_data = parseCSV("data/daily_adjusted_FB.csv")

    #Initialize weights
    gen_weights = np.random.normal(loc=np.pi, scale=eps, size=(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER))
    disc_weights = np.random.normal(loc=0.0, scale=eps, size=(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER))

    #Initialize optimizer
    opt = GradientDescentOptimizer(0.1)
    for i in range(NUM_EPOCHS):



    print("Training the discriminator.")
    for it in range(50):
        disc_weights = opt.step(disc_cost, disc_weights)
        cost = disc_cost(gen_weights, disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it+1, cost))

    print("Training the generator.df")

    # train generator
    for it in range(200):
        gen_weights = opt.step(gen_cost, gen_weights)
        cost = -gen_cost(gen_weights, disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it, cost))



if __name__ == '__main__':
    main()
