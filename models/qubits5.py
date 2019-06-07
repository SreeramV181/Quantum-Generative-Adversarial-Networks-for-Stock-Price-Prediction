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
    #theta_g = theta_g.reshape(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER)

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

# def generator(x, theta_g):
#     """
#     Variational circuit meant to generate next stock price given 4 previous prices
#
#     Args:
#         x: array containing previous 4 stock prices
#         w: variables of the circuit to optimize
#     """
#
#     # hi
#
#     # for i in range(0, NUM_QUBITS - 2):
#     #     qml.CNOT(wires=[i, i + 1])
#     # for i in range(0, NUM_QUBITS - 1):
#     #     qml.Hadamard(wires=i)
#
#     # #Apply a layer of RX
#     # for i in range(0, NUM_QUBITS - 1):
#     #     qml.RX(w[i] * x[i], wires=i)
#
#     # initial_guess_theta = np.random.uniform(low=0, high=2 * np.pi, size=(NUM_QUBITS, NUM_LAYERS, 2))
#     gen_ansatz(x, theta_g)
#
# def discriminator(x, theta_d):
#     """
#     Variational circuit that predicts next stock price based on previous 4 stock prices
#
#     Args:
#         x: array containing previous 5 stock prices
#         w: variables of the circuit to optimize
#     """
#     #Entangle qubits,
#
#     # for i in range(0, NUM_QUBITS):
#     #     qml.Hadamard(wires=i)
#
#     # #Apply a layer of RX
#     # for i in range(0, NUM_QUBITS):
#     #     qml.RX(w[i] * x[i], wires=i)
#     disc_ansatz(x, theta_d)


@qml.qnode(dev)
def real_disc_circuit(data, disc_weights):
    """
    Feeds discriminator with true examples

    """
    disc_ansatz(data, disc_weights)
    return [qml.expval.Hadamard(i) for i in range(NUM_FEATURES + 1)]


@qml.qnode(dev)
def real_gen_circuit(data, gen_weights):
    """
    Feeds discriminator with true examples
    """
    gen_ansatz(data, gen_weights)
    return [qml.expval.Hadamard(i) for i in range(NUM_QUBITS)]

def gen_output(measurements):
    return np.sum([measurements[i] * 2**i for i in range(NUM_QUBITS)])

def prob_real(data):
    return np.sum(data)/(NUM_FEATURES + 1)

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
    print("I'm here")


if __name__ == '__main__':
    main()
