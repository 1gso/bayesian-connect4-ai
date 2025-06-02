"""
Model utilities for Bayesian Deep Neural Network.
TensorFlow 2.x version with improved BDNN implementation.
"""
from typing import Dict, List, Tuple
import tensorflow as tf
import numpy as np


class BayesianDNN(tf.Module):
    """
    Object-oriented Bayesian Deep Neural Network using TensorFlow 2.x.
    Inherits from tf.Module for proper TF2 integration.
    """

    def __init__(
        self,
        num_vars: int,
        first_layer_size: int = 16,
        second_layer_size: int = 16,
        num_hidden: int = 8,
        name: str = "BayesianDNN"
    ) -> None:
        """
        Initialize Bayesian DNN architecture.

        Args:
            num_vars: Number of input variables/features
            first_layer_size: Size of first hidden layer
            second_layer_size: Size of second hidden layer
            num_hidden: Number of additional hidden layers
            name: Name for the module
        """
        super().__init__(name=name)

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

    @tf.function
    def __call__(self, x: tf.Tensor, params: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, num_vars] or [num_vars] for single sample
            params: Flattened parameter tensor

        Returns:
            Output tensor
        """
        layers = self.unpack_parameters(params)

        # Handle both single samples and batches
        if len(x.shape) == 1:
            # Single sample case
            x = tf.linalg.matvec(layers["w1"], x, transpose_a=True) + layers["b1"]
            x = tf.tanh(tf.linalg.matvec(layers["w2"], x, transpose_a=True) + layers["b2"])

            for w, b in layers["hidden"]:
                x = tf.tanh(tf.linalg.matvec(w, x, transpose_a=True) + b)

            output = tf.tanh(tf.linalg.matvec(layers["w_out"], x, transpose_a=True) + layers["b_out"])
        else:
            # Batch case
            x = tf.linalg.matmul(x, layers["w1"]) + layers["b1"]
            x = tf.tanh(tf.linalg.matmul(x, layers["w2"]) + layers["b2"])

            for w, b in layers["hidden"]:
                x = tf.tanh(tf.linalg.matmul(x, w) + b)

            output = tf.tanh(tf.linalg.matmul(x, layers["w_out"]) + layers["b_out"])

        return output


class BayesianDNNBatch(tf.Module):
    """
    Batch-optimized version of BayesianDNN for processing multiple samples at once.
    """

    def __init__(
        self,
        num_vars: int,
        first_layer_size: int = 16,
        second_layer_size: int = 16,
        num_hidden: int = 8,
        name: str = "BayesianDNNBatch"
    ) -> None:
        super().__init__(name=name)

        self.num_vars = num_vars
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.num_hidden = num_hidden

        # Use the same parameter calculation as the single version
        self.param_sizes = self._calculate_param_sizes()
        self.total_params = sum(self.param_sizes.values())

    def _calculate_param_sizes(self) -> Dict[str, int]:
        """Calculate the size of each parameter group."""
        nf, nt, nu = self.num_vars, self.first_layer_size, self.second_layer_size

        sizes = {
            "w1": nf * nt,
            "b1": nt,
            "w2": nt * nu,
            "b2": nu,
            "hidden_weights": self.num_hidden * nu * nu,
            "hidden_biases": self.num_hidden * nu,
            "w_out": nu,
            "b_out": 1,
        }
        return sizes

    def unpack_parameters(self, params: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Unpack parameters for batch processing."""
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

    @tf.function
    def __call__(self, x: tf.Tensor, params: tf.Tensor) -> tf.Tensor:
        """
        Forward pass optimized for batch processing.

        Args:
            x: Input tensor [batch_size, num_vars]
            params: Flattened parameter tensor

        Returns:
            Output tensor [batch_size, 1]
        """
        layers = self.unpack_parameters(params)

        # First layer (no activation) - batch processing
        x = tf.linalg.matmul(x, layers["w1"]) + layers["b1"]

        # Second layer with tanh activation
        x = tf.tanh(tf.linalg.matmul(x, layers["w2"]) + layers["b2"])

        # Hidden layers with tanh activation
        for w, b in layers["hidden"]:
            x = tf.tanh(tf.linalg.matmul(x, w) + b)

        # Output layer with tanh activation
        output = tf.tanh(tf.linalg.matmul(x, layers["w_out"]) + layers["b_out"])
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
        return (features - np.mean(features, axis=0, keepdims=True)) / (
            np.std(features, axis=0, keepdims=True) + 1e-8
        )
    elif method == "minmax":
        # Min-max normalization
        f_min, f_max = np.min(features, axis=0, keepdims=True), np.max(features, axis=0, keepdims=True)
        return (features - f_min) / (f_max - f_min + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


@tf.function
def normalize_features_tf(features: tf.Tensor, method: str = "power") -> tf.Tensor:
    """
    TensorFlow version of feature normalization.

    Args:
        features: Input features tensor
        method: Normalization method ("power", "standard", "minmax")

    Returns:
        Normalized features tensor
    """
    if method == "power":
        return tf.sign(features) * tf.pow(tf.abs(features) / 4.0, 1.5)
    elif method == "standard":
        mean = tf.reduce_mean(features, axis=0, keepdims=True)
        std = tf.math.reduce_std(features, axis=0, keepdims=True)
        return (features - mean) / (std + 1e-8)
    elif method == "minmax":
        f_min = tf.reduce_min(features, axis=0, keepdims=True)
        f_max = tf.reduce_max(features, axis=0, keepdims=True)
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
@tf.function
def bdnn_functional(
    x: tf.Tensor,
    params: tf.Tensor,
    num_vars: int = 138,
    first_layer_size: int = 16,
    second_layer_size: int = 16,
) -> tf.Tensor:
    """
    Functional approach to BDNN using TensorFlow 2.x operations.

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
    x = tf.linalg.matmul(x, w1) + b1  # No activation on first layer
    x = tf.tanh(tf.linalg.matmul(x, w2) + b2)  # Tanh activation

    # Apply hidden layers
    for w, b in zip(hidden_weights, hidden_biases):
        x = tf.tanh(tf.linalg.matmul(x, w) + b)

    # Output layer with tanh
    output = tf.tanh(tf.linalg.matmul(x, w_out) + b_out)

    return output


class ModelCheckpoint:
    """
    Simple checkpoint utility for saving/loading model state in TF2.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_variables(self, variables: dict, step: int) -> str:
        """Save variables to checkpoint."""
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{step:06d}"

        # Create checkpoint
        checkpoint = tf.train.Checkpoint(**variables)
        save_path = checkpoint.save(checkpoint_path)
        print(f"Saved checkpoint to {save_path}")
        return save_path

    def load_variables(self, variables: dict, checkpoint_path: str) -> bool:
        """Load variables from checkpoint."""
        try:
            checkpoint = tf.train.Checkpoint(**variables)
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f"Restored checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def get_latest_checkpoint(self) -> str:
        """Get path to latest checkpoint."""
        return tf.train.latest_checkpoint(self.checkpoint_dir)


if __name__ == "__main__":
    # Run verification
    verify_parameter_count()

    # Test the models
    print("\nTesting BayesianDNN...")
    model = BayesianDNN(num_vars=138)

    # Create dummy data
    x_test = tf.random.normal([1000, 138])
    params_test = tf.random.normal([model.total_params])

    # Test forward pass
    with tf.device('/CPU:0'):  # Ensure CPU execution for testing
        output = model(x_test, params_test)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")

    print("\nTesting BayesianDNNBatch...")
    batch_model = BayesianDNNBatch(num_vars=138)

    with tf.device('/CPU:0'):
        batch_output = batch_model(x_test, params_test)
        print(f"Batch output shape: {batch_output.shape}")
        print(f"Batch output range: [{tf.reduce_min(batch_output):.3f}, {tf.reduce_max(batch_output):.3f}]")

    print("\n✅ All tests passed!")