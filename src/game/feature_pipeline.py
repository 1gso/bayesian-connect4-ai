#!/usr/bin/env python3
"""
Efficient feature pipeline for Connect 4 Q-learning with 300M+ samples.
Uses HDF5 for storage and provides streaming data loaders for training.
"""
import h5py
import numpy as np
from typing import Iterator, Tuple, Optional, List, Dict
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.game.board_processor import BoardProcessor
from src.game.feature_generator import FeatureGenerator


class Connect4FeatureStore:
    """Efficient HDF5-based storage for Connect 4 features and Q-values."""
    
    def __init__(self, filepath: str, mode: str = 'r'):
        """
        Initialize feature store.
        
        Args:
            filepath: Path to HDF5 file
            mode: 'w' for write, 'r' for read, 'a' for append
        """
        self.filepath = filepath
        self.mode = mode
        self.chunk_size = 10000  # Optimal for SSD access
        
    def create_empty_store(self, estimated_samples: int = 300_000_000):
        """Create empty HDF5 file with proper structure."""
        with h5py.File(self.filepath, 'w') as f:
            # Main datasets
            f.create_dataset(
                'features',
                shape=(0, 138),
                maxshape=(None, 138),
                chunks=(self.chunk_size, 138),
                dtype='float32',
                compression='gzip',
                compression_opts=1  # Fast compression
            )
            
            f.create_dataset(
                'q_values',
                shape=(0,),
                maxshape=(None,),
                chunks=(self.chunk_size,),
                dtype='float32'
            )
            
            # Metadata for tracking
            f.create_dataset(
                'game_ids',
                shape=(0,),
                maxshape=(None,),
                chunks=(self.chunk_size,),
                dtype='int32'
            )
            
            f.attrs['total_samples'] = 0
            f.attrs['total_games'] = 0
            
    def append_batch(self, features: np.ndarray, q_values: np.ndarray, 
                    game_ids: np.ndarray):
        """Append a batch of samples efficiently."""
        with h5py.File(self.filepath, 'a') as f:
            # Get current size
            current_size = f['features'].shape[0]
            new_size = current_size + features.shape[0]
            
            # Resize datasets
            f['features'].resize(new_size, axis=0)
            f['q_values'].resize(new_size, axis=0)
            f['game_ids'].resize(new_size, axis=0)
            
            # Write data
            f['features'][current_size:new_size] = features
            f['q_values'][current_size:new_size] = q_values
            f['game_ids'][current_size:new_size] = game_ids
            
            # Update metadata
            f.attrs['total_samples'] = new_size
            f.attrs['total_games'] = int(np.max(game_ids)) + 1
            
    def get_batch(self, batch_size: int, sequential: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of training data.
        
        Args:
            batch_size: Number of samples
            sequential: If False (default), random sampling. If True, sequential from start.
            
        Returns:
            (features, q_values) arrays
        """
        with h5py.File(self.filepath, 'r') as f:
            total_samples = f.attrs['total_samples']
            
            if sequential:
                # For sequential, just get first batch_size samples
                # Could be extended with an offset parameter if needed
                end_idx = min(batch_size, total_samples)
                features = f['features'][0:end_idx]
                q_values = f['q_values'][0:end_idx]
            else:
                # Random sampling for SGD
                indices = np.random.randint(0, total_samples, batch_size)
                features = f['features'][indices]
                q_values = f['q_values'][indices]
            
        return features, q_values


class ParallelPositionProcessor:
    """Process position codes in parallel for maximum throughput."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or mp.cpu_count()
        
    @staticmethod
    def process_game_batch(codes: List[str], game_start_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a batch of games (runs in worker process)."""
        # Import here to avoid pickling issues in multiprocessing
        import sys
        import os
        # Add src to path if needed
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Import the PositionCodeProcessor we already have
        from src.game.position_processor import PositionCodeProcessor
        
        # Each worker creates its own processor instance
        processor = PositionCodeProcessor()
        
        all_features = []
        all_q_values = []
        all_game_ids = []
        
        for idx, code in enumerate(codes):
            game_id = game_start_id + idx
            
            # Use existing process_position_code method
            samples = processor.process_position_code(code, discount=0.95, verbose=False)
            
            # Extract data from samples
            for features, q_value, metadata in samples:
                all_features.append(features)
                all_q_values.append(q_value)
                all_game_ids.append(game_id)
        
        if all_features:
            return (np.array(all_features, dtype=np.float32),
                    np.array(all_q_values, dtype=np.float32),
                    np.array(all_game_ids, dtype=np.int32))
        else:
            return (np.empty((0, 138), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))
    
    def process_position_codes_to_hdf5(self, codes_file: str, output_file: str,
                                     batch_size: int = 1000):
        """Process position codes in parallel and save to HDF5."""
        # Read all codes
        with open(codes_file, 'r') as f:
            all_codes = [line.strip() for line in f if line.strip()]
        
        total_games = len(all_codes)
        print(f"Processing {total_games} games with {self.num_workers} workers...")
        
        # Create output store
        store = Connect4FeatureStore(output_file, mode='w')
        store.create_empty_store()
        
        # Process in batches
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for start_idx in range(0, total_games, batch_size):
                end_idx = min(start_idx + batch_size, total_games)
                batch_codes = all_codes[start_idx:end_idx]
                
                future = executor.submit(
                    self.process_game_batch,
                    batch_codes,
                    start_idx
                )
                futures.append(future)
            
            # Process results as they complete
            processed_games = 0
            total_samples = 0
            
            with tqdm(total=total_games, desc="Processing games") as pbar:
                for future in as_completed(futures):
                    features, q_values, game_ids = future.result()
                    
                    if features.shape[0] > 0:
                        store.append_batch(features, q_values, game_ids)
                        total_samples += features.shape[0]
                    
                    processed_games += batch_size
                    pbar.update(batch_size)
                    pbar.set_postfix({'samples': total_samples})
        
        print(f"\nProcessing complete!")
        print(f"Total games: {total_games}")
        print(f"Total samples: {total_samples}")
        print(f"Average samples per game: {total_samples / total_games:.1f}")
        print(f"Output file: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1e9:.1f} GB")


def main():
    """Process Connect 4 position codes into HDF5 feature store."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Connect 4 feature store")
    parser.add_argument("--input", type=str, required=True,
                       help="Input file with position codes")
    parser.add_argument("--output", type=str, default="connect4_features.h5",
                       help="Output HDF5 file")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Games per batch")
    parser.add_argument("--test-loading", action="store_true",
                       help="Test data loading after processing")
    
    args = parser.parse_args()
    
    # Process games in parallel
    processor = ParallelPositionProcessor(num_workers=args.workers)
    processor.process_position_codes_to_hdf5(
        args.input,
        args.output,
        batch_size=args.batch_size
    )
    
    # Test loading
    if args.test_loading:
        print("\nTesting data loading...")
        store = Connect4FeatureStore(args.output, mode='r')
        
        # Test random batch
        features, q_values = store.get_batch(1024, sequential=False)
        print(f"Random batch shape: {features.shape}")
        print(f"Q-values range: [{q_values.min():.3f}, {q_values.max():.3f}]")
        
        # Test sequential batch
        features, q_values = store.get_batch(1024, sequential=True)
        print(f"Sequential batch shape: {features.shape}")


if __name__ == "__main__":
    main()
