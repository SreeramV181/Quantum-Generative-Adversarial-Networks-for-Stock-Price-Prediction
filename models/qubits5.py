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

    #Entangle qubits,
    for i in range(0, NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i+1])

    #Apply a layer of RX
    for i in range(0, NUM_QUBITS):
        qml.RX(w[0], wires=i)

def discriminator(w):
    """
    Variational circuit that predicts next stock price based on previous 4 stock prices

    Args:
        x: array containing previous 5 stock prices
        w: variables of the circuit to optimize
    """
    #Entangle qubits,
    for i in range(0, NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i+1])

    #Apply a layer of RX
    for i in range(0, NUM_QUBITS):
        qml.RX(w[0], wires=i)

@qml.qnode(dev)
def real_disc_circuit():
    """
    Feeds discriminator with true examples

    """
