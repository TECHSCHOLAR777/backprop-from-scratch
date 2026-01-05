import numpy as np

# --- Activations ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    # derivative wrt activated output
    return a * (1 - a)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# --- Initialization ---
def initial_parameters(i, j):
    return np.random.randn(i, j) * 0.01  # small values

def activation_value(Z, activation_function):
    if activation_function == 'relu':
        return relu(Z)
    elif activation_function == 'sigmoid':
        return sigmoid(Z)

# --- Backprop ---
def backpropagator(X, y, learning_rate, iterations, layers, neurons_array, activation_functions):
    np.random.seed(777)
    param = {}

    # Initialize parameters
    for layer in range(layers):
        if layer == 0:
            W = initial_parameters(X.shape[1], neurons_array[layer])
        else:
            W = initial_parameters(neurons_array[layer-1], neurons_array[layer])
        b = np.zeros((1, neurons_array[layer]))
        param[layer] = {"W": W, "b": b}

    # Training loop
    for i in range(iterations):
        activation_values = []
        Z_values = []

        # Forward pass
        for layer in range(layers):
            W = param[layer]["W"]
            b = param[layer]["b"]
            if layer == 0:
                Z = np.dot(X, W) + b
            else:
                Z = np.dot(activation_values[layer-1], W) + b
            A = activation_value(Z, activation_functions[layer])
            activation_values.append(A)
            Z_values.append(Z)

        # Loss (Binary Cross-Entropy)
        m = X.shape[0]
        A_final = activation_values[-1]
        eps = 1e-8#>>>>>>>>>>> to avoid log(0)
        loss = - (y * np.log(A_final + eps) + (1 - y) * np.log(1 - A_final + eps))
        loss = np.sum(loss) / m
        print("Iteration", i, "Loss:", loss)

        # Backward pass
        dZ = A_final - y  # for sigmoid + BCE
        for layer in reversed(range(layers)):
            A_prev = X if layer == 0 else activation_values[layer-1]
            W = param[layer]["W"]

            m = A_prev.shape[0]
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            print("Layer", layer, "grad norm:", np.linalg.norm(dW))
            print("Layer", layer, "grad norm:", np.linalg.norm(db))

            # Update parameters
            param[layer]["W"] -= learning_rate * dW
            param[layer]["b"] -= learning_rate * db

            if layer > 0:
                if activation_functions[layer-1] == 'sigmoid':
                    dA = np.dot(dZ, W.T)
                    dZ = dA * sigmoid_derivative(activation_values[layer-1])
                elif activation_functions[layer-1] == 'relu':
                    dA = np.dot(dZ, W.T)
                    dZ = dA * relu_derivative(Z_values[layer-1])

    return param   
