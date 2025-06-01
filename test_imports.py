#!/usr/bin/env python3
"""
Test imports to verify project structure.
Run this from the project root to test if imports work.
"""
import sys
import os

# Add src to path
sys.path.insert(0, "src")

print("Testing imports...")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    from game.board_processor import BoardProcessor

    print("✅ BoardProcessor imported successfully")
except ImportError as e:
    print(f"❌ BoardProcessor import failed: {e}")

try:
    from game.feature_generator import FeatureGenerator

    print("✅ FeatureGenerator imported successfully")
except ImportError as e:
    print(f"❌ FeatureGenerator import failed: {e}")

try:
    from training.model_utils import BayesianDNN, verify_parameter_count

    print("✅ Model utils imported successfully")
except ImportError as e:
    print(f"❌ Model utils import failed: {e}")

try:
    from training.bayesian_trainer import CPUGPUBayesianTrainer

    print("✅ Bayesian trainer imported successfully")
except ImportError as e:
    print(f"❌ Bayesian trainer import failed: {e}")

print("\nTesting basic functionality...")
try:
    board = BoardProcessor()
    print("✅ BoardProcessor instantiated")

    model = BayesianDNN(138, 16, 16)
    print(f"✅ BayesianDNN instantiated (params: {model.total_params})")

    verify_parameter_count()
    print("✅ Parameter verification completed")

except Exception as e:
    print(f"❌ Functionality test failed: {e}")

print("\n🎉 Import tests completed!")
