#!/usr/bin/env python3
"""
Training script for Bayesian Connect 4 AI.
TensorFlow 2.x version with improved command-line interface.
"""
import os
import sys
import argparse
from pathlib import Path
import logging

# Ensure we can import from src
if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "src"))

from src.training.bayesian_trainer_tf2 import TF2BayesianTrainer
from src.training.model_utils_tf2 import verify_parameter_count


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )


def check_tensorflow_setup() -> None:
    """Check TensorFlow setup and available devices."""
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    # Check for GPUs
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                # Get GPU memory info if available
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                print(f"    Memory: {memory_info}")
            except:
                pass
    else:
        print("No GPUs found, using CPU")

    # Check CPU info
    print(
        f"CPU threads available: {tf.config.threading.get_inter_op_parallelism_threads()}"
    )


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Check for input file pattern
    expected_file = os.path.join(args.data_dir, f"r139_{args.run_index}.txt")
    if not os.path.exists(expected_file):
        print(f"⚠️  Warning: Expected input file not found: {expected_file}")
        print("Training will fail if this file is missing.")

    # Validate hyperparameters
    if args.leapfrog_steps <= 0:
        raise ValueError("Leapfrog steps must be positive")

    if args.burnin_steps < 0:
        raise ValueError("Burn-in steps must be non-negative")

    if args.num_results <= 0:
        raise ValueError("Number of results must be positive")


def main():
    """Main training function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian Deep Neural Network using TensorFlow 2.x",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "run_index", type=str, help="Unique identifier for this training run"
    )

    parser.add_argument(
        "leapfrog_steps", type=int, help="Number of leapfrog steps for HMC sampling"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing input data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory for saving model outputs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--burnin-steps", type=int, default=5000, help="Number of burn-in steps for HMC"
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=2,
        help="Number of samples to collect after burn-in",
    )

    parser.add_argument(
        "--step-size", type=float, default=0.01, help="Initial step size for HMC"
    )

    parser.add_argument(
        "--prior-scale",
        type=float,
        default=200.0,
        help="Prior scale for parameter initialization",
    )

    parser.add_argument(
        "--likelihood-scale",
        type=float,
        default=0.3,
        help="Scale parameter for likelihood",
    )

    # Model architecture
    parser.add_argument(
        "--first-layer-size", type=int, default=16, help="Size of first hidden layer"
    )

    parser.add_argument(
        "--second-layer-size", type=int, default=16, help="Size of second hidden layer"
    )

    parser.add_argument(
        "--num-hidden", type=int, default=8, help="Number of additional hidden layers"
    )

    # Utility arguments
    parser.add_argument(
        "--verify-params",
        action="store_true",
        help="Verify parameter count before training",
    )

    parser.add_argument(
        "--check-tf", action="store_true", help="Check TensorFlow setup before training"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output and debug logging"
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )

    # Performance arguments
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (experimental)",
    )

    parser.add_argument(
        "--memory-growth",
        action="store_true",
        default=True,
        help="Enable GPU memory growth",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print configuration
    print("🚀 Bayesian Neural Network Training (TensorFlow 2.x)")
    print("=" * 60)
    print(f"Run Index: {args.run_index}")
    print(f"Leapfrog Steps: {args.leapfrog_steps}")
    print(f"Burn-in Steps: {args.burnin_steps}")
    print(f"Number of Results: {args.num_results}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(
        f"Model Architecture: {args.first_layer_size} → {args.second_layer_size} → ({args.num_hidden} hidden) → 1"
    )
    print("=" * 60)

    try:
        # Validate arguments
        validate_arguments(args)
        logger.info("Arguments validated successfully")

        # Check TensorFlow setup if requested
        if args.check_tf:
            print("\n🔍 Checking TensorFlow setup...")
            check_tensorflow_setup()
            print()

        # Verify parameter count if requested
        if args.verify_params:
            print("\n🔍 Verifying parameter count...")
            verify_parameter_count(
                num_vars=138,  # Fixed for this application
                first_layer_size=args.first_layer_size,
                second_layer_size=args.second_layer_size,
            )
            print()

        # Configure mixed precision if requested
        if args.mixed_precision:
            print("⚡ Enabling mixed precision training...")
            import tensorflow as tf

            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_checkpoints:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Initialize trainer with custom parameters
        trainer = TF2BayesianTrainer(
            run_index=args.run_index,
            leapfrog_steps=args.leapfrog_steps,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )

        # Override default hyperparameters if specified
        if args.burnin_steps != 5000:
            trainer.num_burnin_steps = args.burnin_steps
        if args.num_results != 2:
            trainer.num_results = args.num_results
        if args.step_size != 0.01:
            trainer.initial_step_size = args.step_size
        if args.prior_scale != 200.0:
            trainer.initial_prior_scale = args.prior_scale
        if args.likelihood_scale != 0.3:
            trainer.likelihood_scale = args.likelihood_scale

        # Override model architecture if specified
        if (
            args.first_layer_size != 16
            or args.second_layer_size != 16
            or args.num_hidden != 8
        ):

            from src.training.model_utils_tf2 import BayesianDNN

            trainer.bdnn_model = BayesianDNN(
                num_vars=trainer.number_vars,
                first_layer_size=args.first_layer_size,
                second_layer_size=args.second_layer_size,
                num_hidden=args.num_hidden,
            )
            trainer.num_model_parameters = trainer.bdnn_model.total_params
            print(f"Updated model parameters: {trainer.num_model_parameters}")

        logger.info("Starting training process")

        # Run training
        trainer.train()

        print("\n✅ Training completed successfully!")
        logger.info("Training completed successfully")

        # Save final checkpoint if requested
        if args.save_checkpoints:
            print("💾 Saving final checkpoint...")
            from src.training.model_utils_tf2 import ModelCheckpoint

            checkpoint = ModelCheckpoint(args.checkpoint_dir)
            checkpoint.save_variables(
                {
                    "p_mean": trainer.p_mean,
                    "p_scale": trainer.p_scale,
                    "step_size": trainer.step_size,
                },
                step=999999,
            )

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logger.warning("Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")

        if args.verbose:
            import traceback

            traceback.print_exc()
            logger.error("Full traceback:", exc_info=True)

        sys.exit(1)

    finally:
        # Clean up (TF2 handles most cleanup automatically)
        logger.info("Cleaning up resources")


if __name__ == "__main__":
    main()
