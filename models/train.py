from models.qubits5 import *
from utils.parser import *
import pennylane as qml
import random
from pennylane import numpy as np
from pennylane.optimize import *
import pickle



# print(gen_ansatz(0, 0))
def main():
    #Constants
    NUM_EPOCHS = 50
    EPS = 1e-2
    MINIBATCH_SIZE = 5

    training_data = parseCSV("data/daily_adjusted_FB.csv")
    training_data = training_data[-50:]

    statistics = {'gen_loss': [], 'disc_loss': []}

    #Initialize weights
    gen_weights = np.random.normal(loc=np.pi/6, scale=EPS, size=(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER))
    disc_weights = np.random.normal(loc=np.pi/6, scale=EPS, size=(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER))

    #Initialize optimizer
    opt = GradientDescentOptimizer(0.001)
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
                        cost -= np.log(D_real) + np.log(1 - D_fake)
                        #print("Disc_cost = {}".format(cost))
                    cost /= MINIBATCH_SIZE
                    return cost

                def gen_cost(g_weights):
                    cost = 0.0
                    for j in range(MINIBATCH_SIZE):
                        G_real = gen_output(real_gen_circuit(g_weights, data=data[j][0]))
                        D_fake = prob_real(real_disc_circuit(disc_weights, data=data[j][0] + [G_real]))
                        cost -= np.log(D_fake)
                        #print("Gen_cost = {}".format(cost))
                    cost /= MINIBATCH_SIZE
                    return cost

                disc_weights = opt.step(disc_cost, disc_weights)
                epoch_d_cost += disc_cost(disc_weights)
                gen_weights = opt.step(gen_cost, gen_weights)
                epoch_g_cost += gen_cost(gen_weights)
            print("Discriminator cost: {}".format(epoch_d_cost))
            print("Generator cost: {}".format(epoch_g_cost))
            statistics['gen_loss'].append(epoch_g_cost)
            statistics['disc_cost'].append(epoch_d_cost)

            file.write("{} {}\n".format(epoch_g_cost, epoch_d_cost))

            f = open('stats' + str(i) + '.pkl', 'w')   # Pickle file is newly created where foo1.py is
            pickle.dump(statistics, f)          # dump data to f
            f.close()



if __name__ == '__main__':
    main()
