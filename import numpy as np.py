import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_layer_sizes, output_dim):
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_dim = output_dim
        self.weights = []
        self.biases = []
        self.num_layers = len(hidden_layer_sizes) + 1

        # Initialize weights and biases
        layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        for i in range(1, self.num_layers + 1):
            weight_shape = (layer_sizes[i], layer_sizes[i-1])
            bias_shape = (layer_sizes[i], 1)
            self.weights.append(np.random.randn(*weight_shape))
            self.biases.append(np.random.randn(*bias_shape))

    def forward(self, X):
        activations = [X]
        outputs = []

        # Compute activations and outputs for hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            a = self._sigmoid(z)
            activations.append(a)
            outputs.append(z)

        # Compute activations and outputs for output layer
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        a = self._softmax(z)
        activations.append(a)
        outputs.append(z)

        return activations, outputs

    def backward(self, X, y, activations, outputs, learning_rate):
        gradients = []
        batch_size = X.shape[1]

        # Compute gradients for output layer
        dZ = activations[-1] - y
        dW = (1 / batch_size) * np.dot(dZ, activations[-2].T)
        db = (1 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
        gradients.append((dW, db))

        # Compute gradients for hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            dA = np.dot(self.weights[i + 1].T, dZ)
            dZ = dA * self._sigmoid_derivative(outputs[i])
            dW = (1 / batch_size) * np.dot(dZ, activations[i].T)
            db = (1 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))

        # Update weights and biases using gradients
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[-(i + 1)][0]
            self.biases[i] -= learning_rate * gradients[-(i + 1)][1]

    def train(self, X, y, learning_rate=0.01, num_epochs=100, batch_size=32):
        num_samples = X.shape[0]

        for epoch in range(num_epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Split data into batches
            num_batches = num_samples // batch_size
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X_shuffled[start:end].T
                y_batch = y_shuffled[start:end].T

                # Forward pass
                activations, outputs = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, activations, outputs, learning_rate)

            # Compute training loss and accuracy
            train_loss, train_accuracy = self.evaluate(X.T, y.T)

            print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.2%}")

    def evaluate(self, X, y):
        activations, _ = self.forward(X)
        y_pred = np.argmax(activations[-1], axis=0)
        y_true = np.argmax(y, axis=0)
        loss = self._cross_entropy_loss(activations[-1], y)
        accuracy = np.mean(y_pred == y_true)
        return loss, accuracy

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-10
        clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(clipped_y_pred))

# Example usage
input_dim = 32 * 32 * 3
hidden_layer_sizes = [100, 100, 50]
output_dim = 10

mlp = MLP(input_dim, hidden_layer_sizes, output_dim)
mlp.train(X_train, Y_train, learning_rate=0.01, num_epochs=100, batch_size=32)
