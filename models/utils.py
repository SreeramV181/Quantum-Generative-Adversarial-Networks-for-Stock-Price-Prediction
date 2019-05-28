import pennylane as qml

def entangle_qubits(NUM_QUBITS):
    for i in range(0, NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i+1])
