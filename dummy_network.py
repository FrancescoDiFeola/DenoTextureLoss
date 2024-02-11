import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple 1-layer network class
class OneLayerNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Set a seed for PyTorch (for GPU if available)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generate synthetic data
    input_size = 10
    output_size = 5
    num_samples = 1000

    # Create random input and target tensors
    inputs = torch.randn(num_samples, input_size)  # shape: 1000x10
    targets = torch.randn(num_samples, output_size)  # shape: 1000x5

    # Define hyperparameters
    learning_rate = 0.01
    num_epochs = 1000

    # Create an instance of the 1-layer network
    model = OneLayerNetwork(input_size, output_size)

    # Define the Mean Squared Error (MSE) loss function
    criterion = nn.MSELoss()

    # Define the optimizer (e.g., stochastic gradient descent)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)

        # Compute the MSE loss
        loss = 0.5 * criterion(outputs, targets) + torch.tensor(1000000.00, requires_grad=True)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss at each epoch
        if (epoch + 1) % 100 == 0:
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Print gradients for each parameter in the model
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # print(f'Gradient - {name}:')
                    print(param.grad.mean())

    # After training, you can use the trained model for predictions:
    test_input = torch.randn(1, input_size)  # Example test input
    predicted_output = model(test_input)
    print("Predicted Output:", predicted_output)
