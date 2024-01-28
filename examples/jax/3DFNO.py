import jax
import jax.numpy as jnp
from jax import random, grad
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from time import time

def FourierLayer(weights_shape):
    """
    A layer applying FFT, a learned weight in Fourier space, and inverse FFT.
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape
        # Initialize the weights for the Fourier space
        k1, k2 = random.split(rng)
        weights = random.normal(k1, weights_shape)
        return output_shape, (weights,)
    
    def apply_fun(params, inputs):
        # Apply FFT
        fft_inputs = jnp.fft.fftn(inputs, axes=[1, 2, 3])
        # Apply learned weights in Fourier space
        weighted_fft = fft_inputs * params[0]
        # Apply inverse FFT
        return jnp.fft.ifftn(weighted_fft, axes=[1, 2, 3]).real

    return init_fun, apply_fun

def FNO3D(layers, width, weights_shape):
    """
    Define a Fourier Neural Operator (FNO) for 3D problems.
    """
    # Initialize a list of layers
    net_layers = [FourierLayer(weights_shape)]

    # Add hidden layers
    for _ in range(layers):
        net_layers.append(Dense(width))
        net_layers.append(Relu)

    # Use stax.serial to combine layers into a single network
    return stax.serial(*net_layers)

# Define the FNO
layers = 4  # Number of hidden layers
width = 64  # Width of each hidden layer
weights_shape = (64, 64, 64, 10)  # Shape of weights in Fourier space
fno = FNO3D(layers, width, weights_shape)

# Initialize FNO parameters
key = random.PRNGKey(0)
input_shape = (-1, 64, 64, 64, 10)  # Batch size, X, Y, Z, Time dimensions
output_shape, params = fno[0](key, input_shape)

# Dummy data
x = random.normal(key, (1, 64, 64, 64, 10))  # Single batch

# Define a simple loss function for demonstration
def loss_fn(params, inputs):
    preds = fno[1](params, inputs)
    return jnp.mean(preds**2)

# Time the forward pass
start_forward = time()
y = fno[1](params, x)
end_forward = time()

# Time the backward pass
start_backward = time()
grads = grad(loss_fn)(params, x)
end_backward = time()

# Time taken for forward and backward passes
time_forward = end_forward - start_forward
time_backward = end_backward - start_backward

print(time_forward, " ", time_backward)
