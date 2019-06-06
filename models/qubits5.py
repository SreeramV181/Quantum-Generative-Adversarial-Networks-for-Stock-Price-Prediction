import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import *

NUM_QUBITS = 5 # Constant determining number of qubits
NUM_LAYERS = 3 # Layer count

# Create quantum computing device on which to perform calculations
dev = qml.device('default.qubit', wires=5)

def ansatz(x, theta):

    theta = theta.reshape(NUM_QUBITS, NUM_LAYERS, 2)
    qs = theta.shape[0]
    layers = theta.shape[1]
    rxrz = theta.shape[2]

    for i in range(layers):

        # hadamard everything
        for q in range(qs):
            qml.Hadamard(wires=q)

        # rx rz everything
        for q in range(qs):
            qml.RX(x[q] * theta[q, i, 0], wires=q)
            qml.RZ(x[q] * theta[q, i, 1], wires=q)

        # entanglement
        for q in range(qs-1):
            qml.CNOT(wires=[q, q + 1])
        

    return pq



def generator(x, theta):
    """
    Variational circuit meant to generate next stock price given 4 previous prices

    Args:
        x: array containing previous 4 stock prices
        w: variables of the circuit to optimize
    """

    x.append(1)

    # hi

    # for i in range(0, NUM_QUBITS - 2):
    #     qml.CNOT(wires=[i, i + 1])
    # for i in range(0, NUM_QUBITS - 1):
    #     qml.Hadamard(wires=i)

    # #Apply a layer of RX
    # for i in range(0, NUM_QUBITS - 1):
    #     qml.RX(w[i] * x[i], wires=i)

    # initial_guess_theta = np.random.uniform(low=0, high=2 * np.pi, size=(NUM_QUBITS, NUM_LAYERS, 2))
    ansatz(x, theta)

def discriminator(x, theta):
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
    ansatz(x, theta)


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
<<<<<<< HEAD
    return qml.expval.Hadamard(wires=[i for i in range(NUM_QUBITS)])

def disc_cost(data, disc_weights, real):
    output = int("".join(str(real_disc_circuit(data, disc_weights)) for x in test_list), 2)
    return (output - real)**2


def gen_cost(data, gen_weights, real):
    output = int("".join(str(real_gen_circuit(data, gen_weights)) for x in test_list), 2)
    return (output - real)**2
=======
    return qml.expval.Hadamard(wires=[i for i in range(NUM_QUBITS - 1)])




# copy pasted code
def prob_real_true(x, disc_weights):
    """Probability that the discriminator guesses correctly on real data.
    Args:
        disc_weights: variables of the discriminator
        x: one single example
    """
    true_disc_output = real_disc_circuit(x, disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    """Probability that the discriminator guesses wrong on fake data.
    Args:
        gen_weights: variables of the generator
        disc_weights: variables of the discriminator
    """
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true # generator wants to minimize this prob


def disc_cost(gen_weights, disc_weights):
    """Cost for the discriminator. Contains two terms: the probability of classifying
    fake data as real, and the probability of classifying real data correctly.
    Args:
        disc_weights: variables of the discriminator
    """
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights) 
    return cost


def gen_cost(gen_weights, disc_weights):
    """Cost for the generator. Contains only the probability of fake data being classified
    as real.
    Args:
        gen_weights: variables of the generator
    """
    return -prob_fake_true(gen_weights, disc_weights)

def main():
    eps = 1e-2
    gen_weights = np.random.normal(loc=np.pi, scale=eps, size=(NUM_QUBITS, NUM_LAYERS, 2))
    disc_weights = np.random.normal(loc=0.0, scale=eps, size=(NUM_QUBITS, NUM_LAYERS, 2))
    opt = GradientDescentOptimizer(0.1)




    print("Training the discriminator.")
    for it in range(50):
        disc_weights = opt.step(disc_cost, disc_weights) 
        cost = disc_cost(gen_weights, disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it+1, cost))

    print("Probability for the discriminator to classify real data correctly: ", prob_real_true(disc_weights))
    print("Probability for the discriminator to classify fake data as real: ", prob_fake_true(gen_weights, disc_weights))

    print("Training the generator.")

    # train generator
    for it in range(200):
        gen_weights = opt.step(gen_cost, gen_weights)
        cost = -gen_cost(gen_weights, disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it, cost))
    
    print("Probability for the discriminator to classify real data correctly: ", prob_real_true(disc_weights))
    print("Probability for the discriminator to classify fake data as real: ", prob_fake_true(gen_weights, disc_weights))

    # should be close to zero at joint optimum
    print("Final cost function value: ", disc_cost(disc_weights))


if __name__ == '__main__':
    main()
>>>>>>> e2ebde79635fdef6230d3967ab2298ac9f3c410d
