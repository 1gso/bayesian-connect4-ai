"""
Model utilities for Bayesian Deep Neural Network.
Contains improved BDNN implementation with better parameter handling.
"""
from typing import Dict, List, Tuple
import tensorflow as tf
import numpy as np


class BayesianDNN:
    """
    Object-oriented Bayesian Deep Neural Network with clean parameter handling.
    """

    def __init__(
        self,
        num_vars: int,
        first_layer_size: int = 16,
        second_layer_size: int = 16,
        num_hidden: int = 8,
    ) -> None:
        """
        Initialize Bayesian DNN architecture.

        Args:
            num_vars: Number of input variables/features
            first_layer_size: Size of first hidden layer
            second_layer_size: Size of second hidden layer
            num_hidden: Number of additional hidden layers
        """
        self.num_vars = num_vars
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.num_hidden = num_hidden

        # Calculate parameter sizes
        self.param_sizes = self._calculate_param_sizes()
        self.total_params = sum(self.param_sizes.values())

        # Store parameter layout for debugging
        self.param_layout = self._create_param_layout()

    def _calculate_param_sizes(self) -> Dict[str, int]:
        """Calculate the size of each parameter group."""
        nf, nt, nu = self.num_vars, self.first_layer_size, self.second_layer_size

        sizes = {
            "w1": nf * nt,  # First layer weights
            "b1": nt,  # First layer biases
            "w2": nt * nu,  # Second layer weights
            "b2": nu,  # Second layer biases
            "hidden_weights": self.num_hidden * nu * nu,  # All hidden layer weights
            "hidden_biases": self.num_hidden * nu,  # All hidden layer biases
            "w_out": nu,  # Output layer weights
            "b_out": 1,  # Output layer bias
        }
        return sizes

    def _create_param_layout(self) -> List[Tuple[str, int, int]]:
        """Create parameter layout for debugging: [(name, start_idx, end_idx), ...]"""
        layout = []
        idx = 0

        for param_name, size in self.param_sizes.items():
            layout.append((param_name, idx, idx + size))
            idx += size

        return layout

    def print_param_layout(self) -> None:
        """Print parameter layout for debugging."""
        print(f"Bayesian DNN Parameter Layout (Total: {self.total_params})")
        print("-" * 50)
        for name, start, end in self.param_layout:
            print(f"{name:15} : [{start:4d}:{end:4d}] (size: {end - start:4d})")

    def unpack_parameters(self, params: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Unpack flattened parameter vector into structured dict.

        Args:
            params: Flattened parameter tensor

        Returns:
            Dictionary of parameter tensors
        """
        nf, nt, nu = self.num_vars, self.first_layer_size, self.second_layer_size

        layers = {}
        idx = 0

        # First layer
        layers["w1"] = tf.reshape(params[idx: idx + self.param_sizes["w1"]], [nf, nt])
        idx += self.param_sizes["w1"]

        layers["b1"] = params[idx: idx + self.param_sizes["b1"]]
        idx += self.param_sizes["b1"]

        # Second layer
        layers["w2"] = tf.reshape(params[idx: idx + self.param_sizes["w2"]], [nt, nu])
        idx += self.param_sizes["w2"]

        layers["b2"] = params[idx: idx + self.param_sizes["b2"]]
        idx += self.param_sizes["b2"]

        # Hidden layers
        layers["hidden"] = []
        for _ in range(self.num_hidden):
            w = tf.reshape(params[idx: idx + nu * nu], [nu, nu])
            idx += nu * nu

            b = params[idx: idx + nu]
            idx += nu

            layers["hidden"].append((w, b))

        # Output layer
        layers["w_out"] = tf.reshape(params[idx: idx + nu], [nu, 1])
        idx += nu

        layers["b_out"] = tf.expand_dims(params[idx], 0)

        return layers

    def __call__(self, x: tf.Tensor, params: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor
            params: Flattened parameter tensor

        Returns:
            Output tensor
        """
        layers = self.unpack_parameters(params)

        # First layer (no activation)
        x = tf.nn.xw_plus_b(x, layers["w1"], layers["b1"])

        # Second layer with tanh activation
        x = tf.tanh(tf.nn.xw_plus_b(x, layers["w2"], layers["b2"]))

        # Hidden layers with tanh activation
        for w, b in layers["hidden"]:
            x = tf.tanh(tf.nn.xw_plus_b(x, w, b))

        # Output layer with tanh activation
        output = tf.tanh(tf.nn.xw_plus_b(x, layers["w_out"], layers["b_out"]))
        return output


def normalize_features(features: np.ndarray, method: str = "power") -> np.ndarray:
    """
    Normalize features for neural network input.

    Args:
        features: Input features array
        method: Normalization method ("power", "standard", "minmax")

    Returns:
        Normalized features
    """
    if method == "power":
        # Improved version of the f^7 normalization
        return np.sign(features) * (np.abs(features) / 4.0) ** 1.5
    elif method == "standard":
        # Standard z-score normalization
        return (features - np.mean(features)) / (np.std(features) + 1e-8)
    elif method == "minmax":
        # Min-max normalization
        f_min, f_max = np.min(features), np.max(features)
        return (features - f_min) / (f_max - f_min + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def verify_parameter_count(
    num_vars: int = 138, first_layer_size: int = 16, second_layer_size: int = 16
) -> None:
    """
    Verify that parameter counting matches the original calculation.

    Args:
        num_vars: Number of input variables
        first_layer_size: Size of first hidden layer
        second_layer_size: Size of second hidden layer
    """
    # Original calculation from your code
    original_count = (
        (num_vars + 1) * first_layer_size
        + second_layer_size * (first_layer_size + 2)
        + 1
        + 8 * second_layer_size * (second_layer_size + 1)
    )

    # New calculation using BayesianDNN
    model = BayesianDNN(num_vars, first_layer_size, second_layer_size)
    new_count = model.total_params

    print("Parameter Count Verification:")
    print(f"Original calculation: {original_count}")
    print(f"BayesianDNN calculation: {new_count}")
    print(f"Match: {original_count == new_count}")

    if original_count != new_count:
        print("\nDetailed breakdown:")
        model.print_param_layout()


# Functional version for backward compatibility
def bdnn_functional(
    x: tf.Tensor,
    params: tf.Tensor,
    num_vars: int = 138,
    first_layer_size: int = 16,
    second_layer_size: int = 16,
) -> tf.Tensor:
    """
    Functional approach to BDNN that's cleaner than the original.

    Args:
        x: Input tensor
        params: Flattened parameter tensor
        num_vars: Number of input variables
        first_layer_size: Size of first hidden layer
        second_layer_size: Size of second hidden layer

    Returns:
        Output tensor
    """
    # Parameter unpacking with clear variable names
    param_idx = 0

    # Layer 1: Input -> First hidden
    w1_end = param_idx + num_vars * first_layer_size
    w1 = tf.reshape(params[param_idx:w1_end], [num_vars, first_layer_size])
    param_idx = w1_end

    b1_end = param_idx + first_layer_size
    b1 = params[param_idx:b1_end]
    param_idx = b1_end

    # Layer 2: First hidden -> Second hidden
    w2_end = param_idx + first_layer_size * second_layer_size
    w2 = tf.reshape(params[param_idx:w2_end], [first_layer_size, second_layer_size])
    param_idx = w2_end

    b2_end = param_idx + second_layer_size
    b2 = params[param_idx:b2_end]
    param_idx = b2_end

    # Hidden layers (8 layers of second_layer_size -> second_layer_size)
    hidden_weights = []
    hidden_biases = []

    for i in range(8):
        w_end = param_idx + second_layer_size * second_layer_size
        w = tf.reshape(params[param_idx:w_end], [second_layer_size, second_layer_size])
        param_idx = w_end

        b_end = param_idx + second_layer_size
        b = params[param_idx:b_end]
        param_idx = b_end

        hidden_weights.append(w)
        hidden_biases.append(b)

    # Output layer
    w_out_end = param_idx + second_layer_size
    w_out = tf.reshape(params[param_idx:w_out_end], [second_layer_size, 1])
    param_idx = w_out_end

    b_out = tf.expand_dims(params[param_idx], 0)

    # Forward pass
    x = tf.nn.xw_plus_b(x, w1, b1)  # No activation on first layer
    x = tf.tanh(tf.nn.xw_plus_b(x, w2, b2))  # Tanh activation

    # Apply hidden layers
    for w, b in zip(hidden_weights, hidden_biases):
        x = tf.tanh(tf.nn.xw_plus_b(x, w, b))

    # Output layer with tanh
    output = tf.tanh(tf.nn.xw_plus_b(x, w_out, b_out))

    return output


if __name__ == "__main__":
    # Run verification
    verify_parameter_count()
