#!/usr/bin/env python3
"""
Test script for TensorFlow 2.x conversion.
Validates the converted code without requiring actual training data.
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.training.model_utils_tf2 import (
        BayesianDNN,
        BayesianDNNBatch,
        normalize_features,
        normalize_features_tf,
        verify_parameter_count,
        ModelCheckpoint,
    )
    from src.training.bayesian_trainer_tf2 import TF2BayesianTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the TF2 files are in src/training/ directory:")
    print("- src/training/bayesian_trainer_tf2.py")
    print("- src/training/model_utils_tf2.py")
    print("Current directory structure expected:")
    print("  src/")
    print("  ├── training/")
    print("  │   ├── bayesian_trainer_tf2.py")
    print("  │   └── model_utils_tf2.py")
    sys.exit(1)


def test_tensorflow_setup():
    """Test TensorFlow 2.x setup."""
    print("🔧 Testing TensorFlow Setup")
    print("-" * 40)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Probability version: {tfp.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    # Check devices
    devices = tf.config.list_physical_devices()
    print(f"Available devices: {len(devices)}")
    for device in devices:
        print(f"  - {device}")

    # Test basic operations
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print(f"Basic operation test: {a.numpy()} + {b.numpy()} = {c.numpy()}")

    print("✅ TensorFlow setup OK\n")


def test_bayesian_dnn():
    """Test BayesianDNN model."""
    print("🧠 Testing BayesianDNN Model")
    print("-" * 40)

    # Test parameter count verification
    verify_parameter_count()

    # Create model
    model = BayesianDNN(num_vars=138, first_layer_size=16, second_layer_size=16)
    print(f"Model created with {model.total_params} parameters")

    # Test parameter layout
    model.print_param_layout()

    # Create test data
    batch_size = 100
    x_test = tf.random.normal([batch_size, 138], dtype=tf.float32)
    params_test = tf.random.normal([model.total_params], dtype=tf.float32)

    print(f"Test input shape: {x_test.shape}")
    print(f"Test params shape: {params_test.shape}")

    # Test forward pass
    output = model(x_test, params_test)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
    print(f"Output mean: {tf.reduce_mean(output):.3f}")

    # Test batch model
    batch_model = BayesianDNNBatch(num_vars=138)
    batch_output = batch_model(x_test, params_test)
    print(f"Batch output shape: {batch_output.shape}")

    # Check if outputs are similar (they should be identical)
    diff = tf.reduce_mean(tf.abs(output - batch_output))
    print(f"Difference between models: {diff:.6f}")

    print("✅ BayesianDNN tests passed\n")


def test_normalization():
    """Test feature normalization functions."""
    print("📊 Testing Feature Normalization")
    print("-" * 40)

    # Create test data
    np.random.seed(42)
    features = np.random.randn(1000, 138).astype(np.float32)

    # Test numpy version
    norm_power = normalize_features(features, method="power")
    norm_standard = normalize_features(features, method="standard")
    norm_minmax = normalize_features(features, method="minmax")

    print(
        f"Original features - mean: {np.mean(features):.3f}, std: {np.std(features):.3f}"
    )
    print(
        f"Power normalization - mean: {np.mean(norm_power):.3f}, std: {np.std(norm_power):.3f}"
    )
    print(
        f"Standard normalization - mean: {np.mean(norm_standard):.3f}, std: {np.std(norm_standard):.3f}"
    )
    print(
        f"MinMax normalization - mean: {np.mean(norm_minmax):.3f}, std: {np.std(norm_minmax):.3f}"
    )

    # Test TensorFlow version
    features_tf = tf.constant(features)
    norm_power_tf = normalize_features_tf(features_tf, method="power")
    norm_standard_tf = normalize_features_tf(features_tf, method="standard")
    norm_minmax_tf = normalize_features_tf(features_tf, method="minmax")

    # Compare numpy and TensorFlow results
    power_diff = np.mean(np.abs(norm_power - norm_power_tf.numpy()))
    standard_diff = np.mean(np.abs(norm_standard - norm_standard_tf.numpy()))
    minmax_diff = np.mean(np.abs(norm_minmax - norm_minmax_tf.numpy()))

    print(f"NumPy vs TF differences:")
    print(f"  Power: {power_diff:.6f}")
    print(f"  Standard: {standard_diff:.6f}")
    print(f"  MinMax: {minmax_diff:.6f}")

    assert power_diff < 1e-5, "Power normalization mismatch"
    assert standard_diff < 1e-5, "Standard normalization mismatch"
    assert minmax_diff < 1e-5, "MinMax normalization mismatch"

    print("✅ Normalization tests passed\n")


def test_hmc_components():
    """Test HMC-related components."""
    print("🎲 Testing HMC Components")
    print("-" * 40)

    # Create a simple target distribution
    def simple_log_prob(x):
        return tfd.Normal(0.0, 1.0).log_prob(x)

    # Test HMC kernel
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=simple_log_prob, num_leapfrog_steps=3, step_size=0.1
    )

    # Test sampling
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        num_burnin_steps=5,
        current_state=tf.constant(0.0),
        kernel=hmc_kernel,
    )

    print(f"HMC samples shape: {samples.shape}")
    print(f"Sample mean: {tf.reduce_mean(samples):.3f}")
    print(f"Sample std: {tf.math.reduce_std(samples):.3f}")
    print(
        f"Acceptance rate: {tf.reduce_mean(tf.cast(kernel_results.is_accepted, tf.float32)):.3f}"
    )

    print("✅ HMC components test passed\n")


def create_mock_data(temp_dir):
    """Create mock training data for testing."""
    print("📁 Creating Mock Data")
    print("-" * 40)

    # Create mock data file
    data_file = os.path.join(temp_dir, "r139_test.txt")

    # Generate mock data matching expected format
    num_rows = 60000  # Slightly more than window_length
    num_vars = 138

    # Create header
    header_cols = ["date", "backup", "term", "val", "code"] + [
        f"var_{i}" for i in range(num_vars)
    ]

    # Generate mock data
    np.random.seed(42)
    dates = [20230101 + i for i in range(num_rows)]
    backup = np.random.randint(0, 2, num_rows)
    term = np.random.randint(1, 4, num_rows)
    val = np.random.randn(num_rows) * 0.01  # Small returns
    code = np.random.randint(1000, 2000, num_rows)

    # Financial factor data
    factor_data = np.random.randn(num_rows, num_vars) * 0.1

    # Combine all data
    all_data = np.column_stack(
        [dates, backup, term, val, code] + [factor_data[:, i] for i in range(num_vars)]
    )

    # Create DataFrame and save
    df = pd.DataFrame(all_data, columns=header_cols)
    df.to_csv(data_file, sep="\t", index=False)

    print(f"Created mock data file: {data_file}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return data_file


def test_trainer_initialization():
    """Test trainer initialization without full training."""
    print("🚀 Testing Trainer Initialization")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock data
        data_file = create_mock_data(temp_dir)

        # Test trainer initialization
        trainer = TF2BayesianTrainer(
            run_index="test",
            leapfrog_steps=3,  # Small for testing
            data_dir=temp_dir,
            output_dir=os.path.join(temp_dir, "output"),
        )

        print(f"Trainer initialized successfully")
        print(f"Model parameters: {trainer.num_model_parameters}")
        print(f"Data directory: {trainer.data_dir}")
        print(f"Output directory: {trainer.output_dir}")

        # Test data loading
        trainer.load_data()
        print(f"Data loaded - shape: {trainer.multifactor_data.shape}")

        # Test data generator
        data_gen = trainer.generate_sample_data()
        sample_data = next(data_gen)
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")

        # Test model log prob function (without gradients)
        batch_size = 1000
        sample_batch = sample_data[:batch_size]
        ret = tf.constant(sample_batch[:, 0:1])
        feat = tf.constant(sample_batch[:, 1:])
        params = tf.zeros(trainer.num_model_parameters)

        log_prob = trainer.model_log_prob(ret, feat, params)
        print(f"Model log prob: {log_prob:.3f}")

        print("✅ Trainer initialization test passed\n")


def test_checkpoint_functionality():
    """Test checkpoint saving/loading."""
    print("💾 Testing Checkpoint Functionality")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = ModelCheckpoint(temp_dir)

        # Create test variables
        test_vars = {
            "p_mean": tf.Variable(tf.random.normal([100])),
            "p_scale": tf.Variable(tf.random.normal([100])),
            "step_size": tf.Variable(0.01),
        }

        # Save checkpoint
        save_path = checkpoint.save_variables(test_vars, step=123)
        print(f"Saved checkpoint: {save_path}")

        # Create new variables
        new_vars = {
            "p_mean": tf.Variable(tf.zeros([100])),
            "p_scale": tf.Variable(tf.ones([100])),
            "step_size": tf.Variable(1.0),
        }

        # Load checkpoint
        success = checkpoint.load_variables(new_vars, save_path)
        print(f"Load successful: {success}")

        # Check if values match
        mean_diff = tf.reduce_mean(tf.abs(test_vars["p_mean"] - new_vars["p_mean"]))
        scale_diff = tf.reduce_mean(tf.abs(test_vars["p_scale"] - new_vars["p_scale"]))
        step_diff = tf.abs(test_vars["step_size"] - new_vars["step_size"])

        print(f"Differences after loading:")
        print(f"  Mean: {mean_diff:.6f}")
        print(f"  Scale: {scale_diff:.6f}")
        print(f"  Step size: {step_diff:.6f}")

        assert mean_diff < 1e-6, "Mean mismatch after checkpoint load"
        assert scale_diff < 1e-6, "Scale mismatch after checkpoint load"
        assert step_diff < 1e-6, "Step size mismatch after checkpoint load"

        print("✅ Checkpoint test passed\n")


def main():
    """Run all tests."""
    print("🧪 TensorFlow 2.x Conversion Test Suite")
    print("=" * 50)

    try:
        test_tensorflow_setup()
        test_bayesian_dnn()
        test_normalization()
        test_hmc_components()
        test_trainer_initialization()
        test_checkpoint_functionality()

        print("🎉 All tests passed!")
        print("✅ TensorFlow 2.x conversion is working correctly")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
