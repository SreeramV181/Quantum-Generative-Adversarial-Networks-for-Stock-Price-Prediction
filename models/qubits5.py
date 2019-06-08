import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *
import random
import os
import sys

NUM_QUBITS = 8 # Constant determining number of qubits
NUM_LAYERS = 3 # Number of layers in ansatz
PARAMS_PER_LAYER = 2 #Number of parameters per layer; varies based on layer architecture
NUM_FEATURES = 4 # Number of previous stock prices used to predict next stock price

# Create quantum computing device on which to perform calculations
dev = qml.device('default.qubit', wires=NUM_QUBITS)

#Defines the architecture used for the generator
def gen_ansatz(theta_g, x=None):
    #Reshape theta so params are easier to access
    #theta_g = theta_g.reshape(NUM_QUBITS, NUM_LAYERS, PARAMS_PER_LAYER)
    for i in range(NUM_LAYERS):
        # Hadamard
        for q in range(NUM_QUBITS):
            qml.Hadamard(wires=q)

        # RX RZ
        for q in range(NUM_QUBITS):
            qml.RX(x[q // 2].val * theta_g[q, i, 0], wires=q)
            qml.RZ(x[q // 2].val * theta_g[q, i, 1], wires=q)

        # Entanglement
        for q in range(NUM_QUBITS-1):
            qml.CNOT(wires=[q, q + 1])

#Defines the architecture used for the discriminator
def disc_ansatz(theta_d, x=None):
    #Reshape theta so params are easier to access
    #theta_d = theta_d.reshape(NUM_FEATURES + 1, NUM_LAYERS, PARAMS_PER_LAYER)
    for i in range(NUM_LAYERS):
        # Hadamard
        for q in range(NUM_FEATURES + 1):
            qml.Hadamard(wires=q)

        # RX RZ
        for q in range(NUM_FEATURES + 1):
            qml.RX(x[q].val * theta_d[q, i, 0], wires=q)
            qml.RZ(x[q].val * theta_d[q, i, 1], wires=q)

        # Entanglement
        for q in range(NUM_FEATURES):
            qml.CNOT(wires=[q, q + 1])

@qml.qnode(dev)
def real_disc_circuit(disc_weights, data=None):
    """
    Feeds discriminator with true examples

    """
    disc_ansatz(disc_weights,x=data)
    return [qml.expval.PauliZ(i) for i in range(NUM_FEATURES + 1)]


@qml.qnode(dev)
def real_gen_circuit(gen_weights, data=None):
    """
    Feeds discriminator with true examples
    """
    gen_ansatz(gen_weights,x=data)
    return [qml.expval.PauliZ(i) for i in range(NUM_QUBITS)]

def gen_output(measurements):
    return np.sum([measurements[i] * 2**i for i in range(NUM_QUBITS)])

def prob_real(data):
    return data.__abs__().sum()/(NUM_FEATURES + 1)

def main():
    print("I'm here")


if __name__ == '__main__':
    main()
