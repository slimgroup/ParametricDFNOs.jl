import torch
import argparse

parser = argparse.ArgumentParser(description="Run a PyTorch script with specified dimensions.")
parser.add_argument('-x', type=int, help='First dimension (e.g., 10000)', required=True)
parser.add_argument('-y', type=int, help='Second dimension (e.g., 250000)', required=True)
args = parser.parse_args()

# Get dimensions from command line arguments
dim1 = args.x
dim2 = args.y

# Function to convert bytes to megabytes
def bytes_to_MB(bytes):
    return bytes / (1024 ** 2)

# Ensure CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))

    # Create a dense weight matrix and an input array
    W = torch.randn(dim1, dim2, device=device, requires_grad=True)
    input_tensor = torch.randn(dim2, device=device)

    for i in range(2):
        print(f"\nIteration {i + 1}")

        # GPU memory usage before the operation
        print("Memory allocated before operation:", bytes_to_MB(torch.cuda.memory_allocated(device)), "MB")
        print("Memory reserved before operation:", bytes_to_MB(torch.cuda.memory_reserved(device)), "MB")

        # Forward pass
        output = W @ input_tensor

        # Compute loss
        L = output.sum()

        # Backward pass
        L.backward()

        # GPU memory usage after the operation
        print("Memory allocated after operation:", bytes_to_MB(torch.cuda.memory_allocated(device)), "MB")
        print("Memory reserved after operation:", bytes_to_MB(torch.cuda.memory_reserved(device)), "MB")

else:
    print("CUDA is not available. Please run this code on a GPU-enabled environment.")
