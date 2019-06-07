from models.qubits5 import *
from utils.parser import *
import pennylane as qml
import random
from pennylane import numpy as np
from pennylane.optimize import *



# print(gen_ansatz(0, 0))
def main():
    #Constants
    NUM_EPOCHS = 50
    EPS = 1e-2
    MINIBATCH_SIZE = 1

    training_data = parseCSV("data/daily_adjusted_FB.csv")

    #Initialize weights
    gen_weights = np.random.normal(loc=np.pi, scale=EPS, size=(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER))
    disc_weights = np.random.normal(loc=0.0, scale=EPS, size=(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER))

    #Initialize optimizer
    opt = GradientDescentOptimizer(0.1)
    for i in range(NUM_EPOCHS):
        epoch_d_cost = 0
        epoch_g_cost = 0

        random.shuffle(training_data)
        for i in range(0, len(training_data), MINIBATCH_SIZE):
            data = training_data[i:i + MINIBATCH_SIZE]

            #Define costs function based on training data
            def disc_cost(d_weights):
                cost = 0.0
                for j in range(MINIBATCH_SIZE):
                    print(data[j])
                    D_real = prob_real(real_disc_circuit(data[j][0] + [data[j][1]], d_weights))
                    G_real = gen_output(real_gen_circuit(data[j][0], gen_weights))
                    D_fake = prob_real(real_disc_circuit(data[j][0] + [G_real], d_weights))
                    cost -= np.log(D_real) + np.log(1 - D_fake)
                cost /= MINIBATCH_SIZE
                return cost

            def gen_cost(g_weights):
                cost = 0.0
                for j in range(MINIBATCH_SIZE):
                    G_real = gen_output(real_gen_circuit(data[j][0], g_weights))
                    D_fake = prob_real(real_disc_circuit(data[j][0] + [G_real], disc_weights))
                    cost -= np.log(D_fake)
                cost /= MINIBATCH_SIZE
                return cost

            disc_weights = opt.step(disc_cost, disc_weights)
            epoch_d_cost += disc_cost(disc_weights)
            gen_weights = opt.step(gen_cost, gen_weights)
            epoch_g_cost += gen_cost(g_weights)
        print("Discriminator cost: {}".format(epoch_d_cost))
        print("Generator cost: {}".format(epoch_g_cost))

if __name__ == '__main__':
    main()
