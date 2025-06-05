#!/usr/bin/env python3
"""
Importance sampling for Connect 4 Q-learning with sequential bootstrapping
and uniqueness weighting to prevent common features from drowning out rare ones.
"""
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
import mmh3  # MurmurHash3 for fast hashing
from tqdm import tqdm
import pickle


class FeatureUniquenessSampler:
    """
    Implements importance sampling based on feature rarity and sequential dependencies.
    """
    
    def __init__(self, hdf5_path: str, hash_size: int = 2**20):
        """
        Initialize sampler with feature statistics.
        
        Args:
            hdf5_path: Path to HDF5 feature store
            hash_size: Size of hash table for feature counting
        """
        self.hdf5_path = hdf5_path
        self.hash_size = hash_size
        
        # Feature statistics
        self.feature_counts = defaultdict(int)
        self.feature_to_samples = defaultdict(set)  # Which samples contain each feature
        self.sample_weights = None
        self.total_samples = 0
        
        # Sequential dependencies
        self.game_boundaries = []  # [(start_idx, end_idx), ...]
        self.sequential_weights = None
        
    def build_feature_index(self, sample_fraction: float = 0.1):
        """
        Build index of feature frequencies and sample mappings.
        Uses hashing trick for memory efficiency with 138-dim features.
        """
        print("Building feature uniqueness index...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_samples = f.attrs['total_samples']
            features_dset = f['features']
            game_ids_dset = f['game_ids']
            
            # Sample subset for initial statistics
            sample_size = int(self.total_samples * sample_fraction)
            sample_indices = np.random.choice(self.total_samples, sample_size, replace=False)
            
            # Build game boundaries
            print("Finding game boundaries...")
            current_game = -1
            game_start = 0
            
            for idx in tqdm(range(self.total_samples), desc="Scanning games"):
                game_id = game_ids_dset[idx]
                if game_id != current_game:
                    if current_game >= 0:
                        self.game_boundaries.append((game_start, idx))
                    game_start = idx
                    current_game = game_id
            self.game_boundaries.append((game_start, self.total_samples))
            
            # Count feature patterns
            print(f"Analyzing {sample_size} samples for feature patterns...")
            for idx in tqdm(sample_indices, desc="Building index"):
                features = features_dset[idx]
                
                # Hash feature patterns (looking for rare combinations)
                feature_hashes = self._extract_feature_patterns(features)
                
                for fhash in feature_hashes:
                    self.feature_counts[fhash] += 1
                    self.feature_to_samples[fhash].add(idx)
        
        # Compute inverse frequency weights
        self._compute_uniqueness_weights()
        
    def _extract_feature_patterns(self, features: np.ndarray) -> List[int]:
        """
        Extract important feature patterns for uniqueness detection.
        Returns list of hashed patterns.
        """
        patterns = []
        
        # 1. Non-zero feature positions (which squares are occupied)
        nonzero_mask = features != 0
        nonzero_positions = np.where(nonzero_mask)[0]
        
        # Hash the pattern of occupied positions
        if len(nonzero_positions) > 0:
            position_hash = mmh3.hash(nonzero_positions.tobytes()) % self.hash_size
            patterns.append(position_hash)
        
        # 2. Critical features (potential wins/blocks)
        # Features close to ±4 indicate critical positions
        critical_features = np.where(np.abs(features) >= 3)[0]
        if len(critical_features) > 0:
            critical_hash = mmh3.hash(critical_features.tobytes()) % self.hash_size
            patterns.append(critical_hash + self.hash_size)  # Offset to separate namespace
        
        # 3. Feature value histogram (game stage indicator)
        hist, _ = np.histogram(features[nonzero_mask], bins=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
        hist_hash = mmh3.hash(hist.tobytes()) % self.hash_size
        patterns.append(hist_hash + 2 * self.hash_size)
        
        # 4. Specific rare patterns (e.g., fork opportunities)
        # Count features with value ±2 (two in a row with space)
        fork_features = np.sum(np.abs(features) == 2)
        if fork_features >= 2:  # Multiple threats
            fork_hash = mmh3.hash(f"fork_{fork_features}".encode()) % self.hash_size
            patterns.append(fork_hash + 3 * self.hash_size)
        
        return patterns
    
    def _compute_uniqueness_weights(self):
        """
        Compute importance weights based on feature rarity.
        Uses inverse frequency weighting with smoothing.
        """
        print("Computing uniqueness weights...")
        
        # Compute IDF-like weights for each pattern
        pattern_weights = {}
        max_count = max(self.feature_counts.values())
        
        for pattern, count in self.feature_counts.items():
            # Inverse frequency with smoothing
            weight = np.log(max_count / (count + 1)) + 1
            pattern_weights[pattern] = weight
        
        # Aggregate weights for each sample
        self.sample_weights = np.ones(self.total_samples, dtype=np.float32)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            features_dset = f['features']
            
            # Process in chunks for memory efficiency
            chunk_size = 10000
            for start in tqdm(range(0, self.total_samples, chunk_size), desc="Computing weights"):
                end = min(start + chunk_size, self.total_samples)
                chunk_features = features_dset[start:end]
                
                for i, features in enumerate(chunk_features):
                    idx = start + i
                    patterns = self._extract_feature_patterns(features)
                    
                    # Sample weight = max weight among its patterns
                    # (emphasize samples with any rare pattern)
                    sample_weight = 1.0
                    for pattern in patterns:
                        if pattern in pattern_weights:
                            sample_weight = max(sample_weight, pattern_weights[pattern])
                    
                    self.sample_weights[idx] = sample_weight
        
        # Normalize weights
        self.sample_weights /= np.mean(self.sample_weights)
        
    def compute_sequential_weights(self, correlation_threshold: float = 0.3):
        """
        Compute weights considering sequential dependencies within games.
        Reduces weight for positions highly correlated with previous positions.
        """
        print("Computing sequential bootstrapping weights...")
        
        self.sequential_weights = np.ones(self.total_samples, dtype=np.float32)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            features_dset = f['features']
            
            for game_start, game_end in tqdm(self.game_boundaries, desc="Processing games"):
                if game_end - game_start < 2:
                    continue
                
                # Load game features
                game_features = features_dset[game_start:game_end]
                
                # Compute position correlations
                for i in range(1, len(game_features)):
                    # Measure similarity to previous positions
                    current_features = game_features[i]
                    
                    # Average correlation with previous positions
                    correlations = []
                    for j in range(max(0, i-3), i):  # Look at last 3 positions
                        prev_features = game_features[j]
                        
                        # Cosine similarity on non-zero features
                        mask = (current_features != 0) | (prev_features != 0)
                        if np.any(mask):
                            corr = np.dot(current_features[mask], prev_features[mask]) / (
                                np.linalg.norm(current_features[mask]) * 
                                np.linalg.norm(prev_features[mask]) + 1e-8
                            )
                            correlations.append(abs(corr))
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        
                        # Reduce weight for highly correlated positions
                        if avg_correlation > correlation_threshold:
                            weight_factor = 1.0 - (avg_correlation - correlation_threshold) / (1 - correlation_threshold)
                            self.sequential_weights[game_start + i] *= weight_factor
        
        # Normalize
        self.sequential_weights /= np.mean(self.sequential_weights)
        
    def get_importance_weights(self) -> np.ndarray:
        """
        Get combined importance weights for all samples.
        """
        if self.sample_weights is None:
            raise ValueError("Must call build_feature_index first")
            
        # Combine uniqueness and sequential weights
        if self.sequential_weights is not None:
            combined_weights = self.sample_weights * self.sequential_weights
        else:
            combined_weights = self.sample_weights
            
        return combined_weights
    
    def sample_batch(self, batch_size: int, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch with importance weighting.
        
        Args:
            batch_size: Number of samples
            temperature: Temperature for sampling (higher = more uniform)
            
        Returns:
            (features, q_values, weights) where weights are importance weights
        """
        weights = self.get_importance_weights()
        
        # Apply temperature
        if temperature != 1.0:
            weights = np.power(weights, 1.0 / temperature)
            
        # Normalize to probabilities
        probs = weights / np.sum(weights)
        
        # Sample indices
        indices = np.random.choice(self.total_samples, batch_size, p=probs, replace=True)
        
        # Get data
        with h5py.File(self.hdf5_path, 'r') as f:
            features = f['features'][indices]
            q_values = f['q_values'][indices]
            
        # Return importance weights for loss scaling
        importance_weights = weights[indices]
        importance_weights = importance_weights / np.mean(importance_weights)  # Normalize
        
        return features, q_values, importance_weights
    
    def get_rarity_analysis(self, top_k: int = 100) -> Dict:
        """
        Analyze most rare and common patterns.
        """
        analysis = {
            'total_patterns': len(self.feature_counts),
            'total_samples': self.total_samples,
            'rarest_patterns': [],
            'most_common_patterns': []
        }
        
        # Sort by count
        sorted_patterns = sorted(self.feature_counts.items(), key=lambda x: x[1])
        
        # Get rarest
        for pattern, count in sorted_patterns[:top_k]:
            analysis['rarest_patterns'].append({
                'pattern_id': pattern,
                'count': count,
                'frequency': count / self.total_samples,
                'weight': np.log(max(self.feature_counts.values()) / (count + 1)) + 1
            })
            
        # Get most common
        for pattern, count in sorted_patterns[-top_k:]:
            analysis['most_common_patterns'].append({
                'pattern_id': pattern,
                'count': count,
                'frequency': count / self.total_samples,
                'weight': np.log(max(self.feature_counts.values()) / (count + 1)) + 1
            })
            
        return analysis
    
    def save_index(self, filepath: str):
        """Save computed index for reuse."""
        index_data = {
            'feature_counts': dict(self.feature_counts),
            'sample_weights': self.sample_weights,
            'sequential_weights': self.sequential_weights,
            'game_boundaries': self.game_boundaries,
            'total_samples': self.total_samples
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
            
    def load_index(self, filepath: str):
        """Load precomputed index."""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.feature_counts = defaultdict(int, index_data['feature_counts'])
        self.sample_weights = index_data['sample_weights']
        self.sequential_weights = index_data['sequential_weights']
        self.game_boundaries = index_data['game_boundaries']
        self.total_samples = index_data['total_samples']


class ImportanceSamplingDataLoader:
    """
    Data loader with importance sampling for training.
    """
    
    def __init__(self, sampler: FeatureUniquenessSampler, 
                 batch_size: int = 1024,
                 temperature_schedule: Optional[callable] = None):
        """
        Initialize loader.
        
        Args:
            sampler: Feature uniqueness sampler
            batch_size: Batch size
            temperature_schedule: Function mapping epoch -> temperature
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.temperature_schedule = temperature_schedule or (lambda epoch: 1.0)
        self.epoch = 0
        
    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get single batch with importance weights."""
        temperature = self.temperature_schedule(self.epoch)
        return self.sampler.sample_batch(self.batch_size, temperature)
        
    def iterate_epoch(self, steps_per_epoch: int):
        """Iterate for one epoch."""
        for _ in range(steps_per_epoch):
            yield self.get_batch()
        self.epoch += 1


def main():
    """Build importance sampling index for Connect 4 features."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build importance sampling index")
    parser.add_argument("--input", type=str, required=True,
                       help="Input HDF5 file")
    parser.add_argument("--output", type=str, default="importance_index.pkl",
                       help="Output index file")
    parser.add_argument("--sample-fraction", type=float, default=0.1,
                       help="Fraction of data to sample for index building")
    parser.add_argument("--analyze", action="store_true",
                       help="Print rarity analysis")
    parser.add_argument("--test-sampling", action="store_true",
                       help="Test importance sampling")
    
    args = parser.parse_args()
    
    # Build index
    sampler = FeatureUniquenessSampler(args.input)
    sampler.build_feature_index(sample_fraction=args.sample_fraction)
    sampler.compute_sequential_weights()
    
    # Save index
    sampler.save_index(args.output)
    print(f"Saved importance sampling index to {args.output}")
    
    # Analyze if requested
    if args.analyze:
        analysis = sampler.get_rarity_analysis()
        print("\nRarity Analysis:")
        print(f"Total unique patterns: {analysis['total_patterns']}")
        print("\nRarest patterns:")
        for p in analysis['rarest_patterns'][:10]:
            print(f"  Pattern {p['pattern_id']}: {p['count']} occurrences, weight={p['weight']:.2f}")
        print("\nMost common patterns:")
        for p in analysis['most_common_patterns'][-10:]:
            print(f"  Pattern {p['pattern_id']}: {p['count']} occurrences, weight={p['weight']:.2f}")
    
    # Test sampling
    if args.test_sampling:
        print("\nTesting importance sampling...")
        
        # Temperature schedule: start high, decrease over time
        temp_schedule = lambda epoch: 2.0 * (0.9 ** epoch)
        
        loader = ImportanceSamplingDataLoader(sampler, batch_size=1024, 
                                            temperature_schedule=temp_schedule)
        
        # Sample a few batches
        for i in range(3):
            features, q_values, weights = loader.get_batch()
            print(f"\nBatch {i}:")
            print(f"  Features shape: {features.shape}")
            print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
            print(f"  Weight std: {weights.std():.3f}")
            
            # Check which samples have high weights
            high_weight_idx = np.argmax(weights)
            print(f"  Highest weight sample has {np.sum(features[high_weight_idx] != 0)} non-zero features")


if __name__ == "__main__":
    main()
