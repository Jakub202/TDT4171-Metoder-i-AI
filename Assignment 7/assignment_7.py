import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.
    # Activation function and its derivative
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(x):
        return x * (1 - x)


    # Initialize network parameters
    np.random.seed(42)  # For consistent initialization
    input_layer_size = X_train.shape[1]
    hidden_layer_size = 2
    output_layer_size = 1
    weights_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))
    bias_hidden = np.random.uniform(-1, 1, (1, hidden_layer_size))
    bias_output = np.random.uniform(-1, 1, (1, output_layer_size))

    learning_rate = 0.001
    epochs = 5000

    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = output_layer_input  # Linear activation for regression

        # Compute error
        error = y_train.reshape(-1, 1) - predicted_output

        # Backpropagation
        d_predicted_output = error
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        # Print the error every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(error))
            print(f'Epoch {epoch}, Loss: {loss}')

    # Testing
    hidden_layer_input = np.dot(X_test, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_test_output = output_layer_input

    # Compute mean squared error on test set
    test_loss = np.mean(np.square(y_test.reshape(-1, 1) - predicted_test_output))
    print(f'Test Loss: {test_loss}')