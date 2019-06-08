import pennylane as qml
from pennylane import numpy as np

N = 2
dev = qml.device('forest.numpy_wavefunction', wires=N)

@qml.qnode(dev)
def qaoa_one_qubit(param, problem=None):
    print(problem)
    print('problem = ', [i.val for i in problem])
    for i in problem:
        qml.RX(param[0], wires=i)
    return qml.expval.PauliZ(i)

problem = [0]
dcircuit = qml.grad(qaoa_one_qubit, argnum=0)
dcircuit([0.5], problem=problem)
