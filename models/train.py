import pennylane as qml
import random
from pennylane import numpy as np
from pennylane.optimize import *
import pickle
import os
import sys
sys.path.append("/Users/sreeramv/Documents/GitHub/Quantum-Generative-Adversarial-Networks-for-Stock-Price-Prediction/")
from models.qubits5 import *
from models.parser import *

# print(gen_ansatz(0, 0))
def main():
    #Constants
    NUM_EPOCHS = 50
    EPS = 1e-1
    MINIBATCH_SIZE = 1
    NUM_DATAPOINTS = 50

    training_data = parseCSV("/Users/sreeramv/Documents/GitHub/Quantum-Generative-Adversarial-Networks-for-Stock-Price-Prediction/data/daily_adjusted_FB.csv")
    training_data = training_data[-NUM_DATAPOINTS:]

    statistics = {'gen_loss': [], 'disc_loss': []}

    #Initialize weights
    gen_weights = np.random.normal(loc=np.pi/7, scale=EPS, size=(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER))
    disc_weights = np.random.normal(loc=np.pi/7, scale=EPS, size=(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER))

    #Initialize optimizer
    opt_gen = RMSPropOptimizer(stepsize=1e-2)
    opt_disc = RMSPropOptimizer(stepsize=1e-2)
    with open("costs.txt", "a+") as file:
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
                        D_real = prob_real(real_disc_circuit(d_weights, data=data[j][0] + [data[j][1]]))
                        G_real = gen_output(real_gen_circuit(gen_weights, data=data[j][0]))
                        D_fake = prob_real(real_disc_circuit(d_weights, data=data[j][0] + [G_real]))
                        if type(D_real) != np.float64:
                            D_real = D_real._value
                        if type(D_fake) != np.float64:
                            D_fake = D_fake._value
                        cost -= np.log(D_real) + np.log(1 - D_fake)
                    cost /= MINIBATCH_SIZE
                    return cost

                def gen_cost(g_weights):
                    cost = 0.0
                    for j in range(MINIBATCH_SIZE):
                        G_real = gen_output(real_gen_circuit(g_weights, data=data[j][0]))
                        D_fake = prob_real(real_disc_circuit(disc_weights, data=data[j][0] + [G_real]))
                        if type(D_fake) != np.float64:
                            D_fake = D_fake._value
                        cost -= np.log(D_fake)
                    cost /= MINIBATCH_SIZE
                    return cost

                disc_grad = opt_disc.compute_grad(disc_cost, disc_weights)
                gen_grad = opt_gen.compute_grad(gen_cost, gen_weights)
                print(disc_grad[:1])
                print(gen_grad[:1])
                disc_weights = opt_disc.apply_grad(disc_grad, disc_weights)
                gen_weights = opt_gen.apply_grad(gen_grad, gen_weights)

                epoch_d_cost += disc_cost(disc_weights)
                epoch_g_cost += gen_cost(gen_weights)

                # disc_weights = opt_disc.step(disc_cost, disc_weights)
                # gen_weights = opt_gen.step(gen_cost, gen_weights)

                print("Curr Disc Cost: {}".format(epoch_d_cost))
                print("Curr Gen Cost: {}".format(epoch_g_cost))
            print("Discriminator cost: {}".format(epoch_d_cost))
            print("Generator cost: {}".format(epoch_g_cost))
            print("\n")
            statistics['gen_loss'].append(epoch_g_cost)
            statistics['disc_loss'].append(epoch_d_cost)

            file.write("{} {}\n".format(epoch_g_cost, epoch_d_cost))

            f = open('stats' + str(i) + '.pkl', 'wb')   # Pickle file is newly created where foo1.py is
            pickle.dump(statistics, f)          # dump data to f
            f.close()

            if i % 10 == 0:
                d = open('disc_weights_' + str(i) + '.pkl', 'wb')   # Pickle file is newly created where foo1.py is
                pickle.dump(disc_weights, d)          # dump data to f
                d.close()

                g = open('gen_weights_' + str(i) + '.pkl', 'wb')   # Pickle file is newly created where foo1.py is
                pickle.dump(gen_weights, g)          # dump data to f
                g.close()



if __name__ == '__main__':
    main()
