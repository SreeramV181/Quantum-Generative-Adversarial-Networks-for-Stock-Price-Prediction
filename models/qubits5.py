import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *

NUM_QUBITS = 5 #Constant determining number of qubits

# Create quantum computing device on which to perform calculations
dev = qml.device('default.qubit', wires=5)

def generator(x, w):
    """
    Variational circuit meant to generate next stock price given 4 previous prices

    Args:
        x: array containing previous 5 stock prices
        w: variables of the circuit to optimize
    """

    x.append(1)

    #
    for i in range(0, NUM_QUBITS - 1):
        qml.Hadamard(wires=i)

    #Apply a layer of RX
    for i in range(0, NUM_QUBITS - 1):
        qml.RX(w[i] * x[i], wires=i)

def discriminator(x, w):
    """
    Variational circuit that predicts next stock price based on previous 4 stock prices

    Args:
        x: array containing previous 5 stock prices
        w: variables of the circuit to optimize
    """
    #Entangle qubits,
    for i in range(0, NUM_QUBITS):
        qml.Hadamard(wires=i)

    #Apply a layer of RX
    for i in range(0, NUM_QUBITS):
        qml.RX(w[i] * x[i], wires=i)

@qml.qnode(dev)
def real_disc_circuit(data, disc_weights):
    """
    Feeds discriminator with true examples

    """
    discriminator(data, disc_weights)
    return qml.expval.Hadamard(wires=[i for i in range(NUM_QUBITS)])

@qml.qnode(dev)
def real_gen_circuit(data, gen_weights):
    """
    Feeds discriminator with true examples

    """
    generator(data, gen_weights)
    return qml.expval.Hadamard(wires=[i for i in range(NUM_QUBITS - 1)])
