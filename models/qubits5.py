import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *
from utils.parser import *
import random

NUM_QUBITS = 8 # Constant determining number of qubits
NUM_LAYERS = 3 # Number of layers in ansatz
PARAMS_PER_LAYER = 2 #Number of parameters per layer; varies based on layer architecture
NUM_FEATURES = 4 # Number of previous stock prices used to predict next stock price

# Create quantum computing device on which to perform calculations
dev = qml.device('default.qubit', wires=NUM_QUBITS)

#Defines the architecture used for the generator
def gen_ansatz(x, theta_g):
    #Reshape theta so params are easier to access
    theta_g = theta_g.reshape(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER)

    for i in range(NUM_LAYERS):
        # Hadamard
        for q in range(NUM_QUBITS):
            qml.Hadamard(wires=q)

        # RX RZ
        for q in range(NUM_QUBITS):
            qml.RX(x[q // 2] * theta_g[q, i, 0], wires=q)
            qml.RZ(x[q // 2] * theta_g[q, i, 1], wires=q)

        # Entanglement
        for q in range(NUM_QUBITS-1):
            qml.CNOT(wires=[q, q + 1])

#Defines the architecture used for the discriminator
def disc_ansatz(x, theta_d):
    #Reshape theta so params are easier to access
    theta_d = theta_d.reshape(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER)

    for i in range(NUM_LAYERS):
        # Hadamard
        for q in range(NUM_FEATURES + 1):
            qml.Hadamard(wires=q)

        # RX RZ
        for q in range(NUM_FEATURES + 1):
            qml.RX(x[q] * theta_d[q, i, 0], wires=q)
            qml.RZ(x[q] * theta_d[q, i, 1], wires=q)

        # Entanglement
        for q in range(NUM_FEATURES):
            qml.CNOT(wires=[q, q + 1])

def generator(x, theta_g):
    """
    Variational circuit meant to generate next stock price given 4 previous prices

    Args:
        x: array containing previous 4 stock prices
        w: variables of the circuit to optimize
    """

    # hi

    # for i in range(0, NUM_QUBITS - 2):
    #     qml.CNOT(wires=[i, i + 1])
    # for i in range(0, NUM_QUBITS - 1):
    #     qml.Hadamard(wires=i)

    # #Apply a layer of RX
    # for i in range(0, NUM_QUBITS - 1):
    #     qml.RX(w[i] * x[i], wires=i)

    # initial_guess_theta = np.random.uniform(low=0, high=2 * np.pi, size=(NUM_QUBITS, NUM_LAYERS, 2))
    gen_ansatz(x, theta_g)

def discriminator(x, theta_d):
    """
    Variational circuit that predicts next stock price based on previous 4 stock prices

    Args:
        x: array containing previous 5 stock prices
        w: variables of the circuit to optimize
    """
    #Entangle qubits,

    # for i in range(0, NUM_QUBITS):
    #     qml.Hadamard(wires=i)

    # #Apply a layer of RX
    # for i in range(0, NUM_QUBITS):
    #     qml.RX(w[i] * x[i], wires=i)
    disc_ansatz(x, theta_d)


@qml.qnode(dev)
def real_disc_circuit(data, disc_weights):
    """
    Feeds discriminator with true examples

    """
    discriminator(data, disc_weights)
    votes = 0
    for i in range(NUM_FEATURES + 1):
        votes += qml.expval.PauliZ(i)
    return votes/5

@qml.qnode(dev)
def real_gen_circuit(data, gen_weights):
    """
    Feeds discriminator with true examples
    """
    generator(data, gen_weights)
    output = 0.0
    for i in range(len(NUM_QUBITS)):
        output += qml.expval.PauliZ(i) * 2**i
    return output

# def disc_cost(data=None, gen_weights=None, disc_weights):
#     D_real = real_disc_circuit(data, disc_weights)
#     G_real = real_gen_circuit(data[:NUM_FEATURES], gen_weights)
#     D_fake = real_disc_circuit(data[:NUM_FEATURES] + [G_real], disc_weights)
#     return -np.log(D_real) - np.log(1 - D_fake)
#
# def gen_cost(data=None, gen_weights, disc_weights=None):
#     G_real = real_gen_circuit(data[:NUM_FEATURES], gen_weights)
#     D_fake = real_disc_circuit(data[:NUM_FEATURES] + [G_real], disc_weights)
#     return -np.log(D_fake)

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
                    D_real = real_disc_circuit(data[j][0] + [data[j][1]], d_weights)
                    G_real = real_gen_circuit(data[j][0], gen_weights)
                    D_fake = real_disc_circuit(data[j][0] + [G_real], d_weights)
                    cost -= np.log(D_real) + np.log(1 - D_fake)
                cost /= MINIBATCH_SIZE
                return cost

            def gen_cost(g_weights):
                cost = 0.0
                for j in range(MINIBATCH_SIZE):
                    G_real = real_gen_circuit(data[j][0], g_weights)
                    D_fake = real_disc_circuit(data[j][0] + [G_real], disc_weights)
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
