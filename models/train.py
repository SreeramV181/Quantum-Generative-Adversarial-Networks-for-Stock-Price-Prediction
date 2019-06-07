import pennylane as qml
import torch
import torch.autograd as Variable
import numpy as np
import random

from models.qubits5 import *
from utils.parser import *



# print(gen_ansatz(0, 0))
def main():
    #Constants
    NUM_EPOCHS = 50
    EPS = 1e-2
    MINIBATCH_SIZE = 1

    training_data = parseCSV("data/daily_adjusted_FB.csv")

    #Initialize weights
    gen_weights = torch.tensor(torch.randn(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER, dtype=torch.float64) * EPS + np.pi, requires_grad=True)
    disc_weights = torch.tensor(torch.randn(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER, dtype=torch.float64) * EPS, requires_grad=True)

    #Initialize optimizer
    optimizer = torch.optim.SGD(params=[gen_weights, disc_weights], lr=.1, momentum=.9)
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
                    D_real = prob_real(real_disc_circuit(d_weights, data=torch.tensor(data[j][0] + [data[j][1]], requires_grad=False)))
                    G_real = gen_output(real_gen_circuit(gen_weights, data=torch.tensor(data[j][0], requires_grad=False)))
                    D_fake = prob_real(real_disc_circuit(d_weights, data=torch.tensor(data[j][0] + [G_real], requires_grad=False)))
                    cost -= np.log(D_real) + np.log(1 - D_fake)
                cost /= MINIBATCH_SIZE
                return cost

            def gen_cost(g_weights):
                cost = 0.0
                for j in range(MINIBATCH_SIZE):
                    G_real = gen_output(real_gen_circuit(g_weights, data=torch.tensor(data[j][0], requires_grad=False)))
                    D_fake = prob_real(real_disc_circuit(disc_weights, data=torch.tensor(data[j][0] + [G_real], requires_grad=False)))
                    cost -= np.log(D_fake)
                cost /= MINIBATCH_SIZE
                return cost

            optimizer.zero_grad()
            disc_loss = disc_cost(disc_weights)
            disc_loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            gen_loss = gen_cost(gen_weights)
            gen_loss.backward()
            optimizer.step()

            epoch_d_cost += disc_loss.item()
            epoch_g_cost += gen_loss.item()
        print("Discriminator cost: {}".format(epoch_d_cost))
        print("Generator cost: {}".format(epoch_g_cost))

if __name__ == '__main__':
    main()
