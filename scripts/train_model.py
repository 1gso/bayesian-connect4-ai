#!/usr/bin/env python3
"""
Training script for Bayesian Connect 4 AI.
Provides a clean command-line interface for training the model.
"""
import os
import sys
import argparse
# from pathlib import Path

# Ensure we can import from src
if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "src"))

from src.training.bayesian_trainer import CPUGPUBayesianTrainer
from src.training.model_utils import verify_parameter_count


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian Deep Neural Network for Connect 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "run_index", type=str, help="Unique identifier for this training run"
    )

    parser.add_argument(
        "leapfrog_steps", type=int, help="Number of leapfrog steps for HMC sampling"
    )

    # Optional arguments
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing input data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory for saving model outputs",
    )

    parser.add_argument(
        "--verify-params",
        action="store_true",
        help="Verify parameter count before training",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Print configuration
    print("🚀 Bayesian Connect 4 AI Training")
    print("=" * 50)
    print(f"Run Index: {args.run_index}")
    print(f"Leapfrog Steps: {args.leapfrog_steps}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 50)

    # Verify parameter count if requested
    if args.verify_params:
        print("\n🔍 Verifying parameter count...")
        verify_parameter_count()
        print()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print("Please ensure your data is in the correct location.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = CPUGPUBayesianTrainer(
        run_index=args.run_index,
        leapfrog_steps=args.leapfrog_steps,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    try:
        # Run training
        trainer.train()
        print("\n✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up
        trainer.close()


if __name__ == "__main__":
    main()
