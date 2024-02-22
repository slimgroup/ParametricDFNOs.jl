import torch
import torch.nn as nn
import argparse
import torch.cuda.memory as mem

# Argument parsing
parser = argparse.ArgumentParser(description='Process xyz and t.')
parser.add_argument('--x', type=int, required=True, help='Dimension x')
parser.add_argument('--y', type=int, required=True, help='Dimension y')
parser.add_argument('--z', type=int, required=True, help='Dimension z')
parser.add_argument('--t', type=int, required=True, help='Dimension t')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Neural Network Model
class SimpleLinearNet(nn.Module):
    def __init__(self):
        super(SimpleLinearNet, self).__init__()
        self.linear1 = nn.Linear(5, 20)
        self.linear2 = nn.Linear(20, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        n, c, *spatial_dims = x.shape
        x = x.view(n, c, -1)  # Reshape to (n, c, xyzt)

        x = self.linear1(x.view(-1, c))  # Flatten spatial dimensions for linear layer
        x = x.view(n, -1, *spatial_dims)  # Reshape back to include spatial dimensions

        x = self.linear2(x.view(n, -1, *spatial_dims).view(-1, 20))
        x = nn.ReLU()(x.view(n, -1, *spatial_dims))

        x = self.linear3(x.view(n, -1, *spatial_dims).view(-1, 128))
        x = nn.ReLU()(x.view(n, -1, *spatial_dims))

        return x

# Initialize model
model = SimpleLinearNet().to(device)

# Create a dummy input and target
x, y, z, t = args.x, args.y, args.z, args.t
input_tensor = torch.randn(1, 5, x, y, z, t, requires_grad=True).to(device)
target = torch.randn(1, 1, x, y, z, t).to(device)

# Forward pass and compute loss
mem_before = mem.memory_allocated(device)
output = model(input_tensor)
loss = torch.norm(output - target) / torch.norm(target)
mem_after = mem.memory_allocated(device)
mem_forward = (mem_after - mem_before) / (1024**2)  # in MB

# Backward pass
# loss.backward()
mem_backward = (mem.memory_allocated(device) - mem_after) / (1024**2)  # in MB

# Print memory consumption
print(f"Memory consumed in forward pass: {mem_forward} MB")
print(f"Memory consumed in backward pass: {mem_backward} MB")
