# Quantum-Generative-Adversarial-Networks-for-Stock-Price-Prediction
This project uses the Pennylane library to create quantum variational circuits capable of predicting a stock's price.

## Instructions to Run

First run

pip install pennylane

pip install pennylane-forest

pip install pickle

and any other imported files.

There are some installation errors with Pennylane. Please follow this link and enter your usr/bin area where Pennylane is installed and follow the recommended fixes. https://discuss.pennylane.ai/t/passing-non-differentiable-arguments-to-qnode/135

Additionally, in train.py change the sys.path.append(input) to be input = {path to the directory} for the modules to work properly. There should be two places where you have to do this.
