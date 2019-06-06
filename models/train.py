from models.qubits5 import *
from utils.parser import *
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *


# print(gen_ansatz(0, 0))
def main():
    print(NUM_FEATURES)
    training = parseCSV("data/daily_adjusted_FB.csv")
    num_epochs = 50

    eps = 1e-2
    gen_weights = np.random.normal(loc=np.pi, scale=eps, size=(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER))
    disc_weights = np.random.normal(loc=0.0, scale=eps, size=(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER))
    opt = GradientDescentOptimizer(0.1)

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

    print("Probability for the discriminator to classify real data correctly: ", prob_real_true(disc_weights))
    print("Probability for the discriminator to classify fake data as real: ", prob_fake_true(gen_weights, disc_weights))

    # should be close to zero at joint optimum
    print("Final cost function value: ", disc_cost(disc_weights))

if __name__ == '__main__':
    main()
