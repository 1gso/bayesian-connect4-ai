#!/usr/bin/env python3
"""
Process Connect 4 position codes into Q-learning training data.
Converts encoded game positions into state-action features with Monte Carlo returns.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from src.game.board_processor import BoardProcessor
from src.game.feature_generator import FeatureGenerator


class PositionCodeProcessor:
    """Process position codes into training data for Bayesian Q-learning."""

    def __init__(self):
        self.board_processor = BoardProcessor()
        self.feature_generator = FeatureGenerator()

    def detect_winner_from_features(self, state_list: List[List[int]]) -> Optional[int]:
        """
        Detect winner using the feature generator's convolution approach.

        Returns: 1 (player 1 wins), -1 (player 2 wins), 0 (draw), None (game ongoing)
        """
        # Get the convolution group matrices
        (
            board,
            group_matrices,
        ) = self.feature_generator.convolution_feature_group_matrices(state_list)

        # Check each feature group for wins (4 or -4 in convolution output)
        for matrix in group_matrices:
            # Player 1 win
            if np.any(matrix == 4):
                return 1
            # Player 2 win
            if np.any(matrix == -4):
                return -1

        # Check for draw (all columns full)
        if all(len(col) == 6 for col in state_list):
            return 0

        return None

    def extract_state_action_features(
        self, state_list: List[List[int]], action: int, player: int
    ) -> np.ndarray:
        """
        Extract features for Q(state, action) from current player's perspective.
        Uses the symmetry of convolution: instead of flipping the board,
        we can just flip the features when player == -1.
        """
        # Get current state features
        _, current_features = self.feature_generator.convolution_feature_gen(state_list)

        # Create copy and apply action
        next_state = [col.copy() for col in state_list]
        next_state[action].append(player)

        # Get next state features
        _, next_features = self.feature_generator.convolution_feature_gen(next_state)

        # If player -1, flip features to normalize perspective
        if player == -1:
            current_features = -current_features
            next_features = -next_features

        # Concatenate
        return np.concatenate([current_features, next_features])

    def process_position_code(
        self, code: str, discount: float = 0.95, verbose: bool = False
    ) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Process a single position code into training samples.

        Args:
            code: Position code like "CM5MLeBCmGoji"
            discount: Discount factor for Monte Carlo returns
            verbose: Print game details

        Returns:
            List of (features, q_value, metadata) tuples
        """
        try:
            # Use BoardProcessor to decode and replay the game
            moves = self.board_processor.decode_moves_code(code)
            # Note: decode_moves_code already calls generate_state_list internally

            if verbose:
                print(f"Code: {code}")
                print(f"Moves: {moves}")

            # Collect all state-action pairs by replaying the game
            training_samples = []
            self.board_processor.reset_state_list()
            player = 1

            for move_idx, action in enumerate(moves):
                # Get current state
                state_list = self.board_processor.state_list

                # Extract features BEFORE making the move
                features = self.extract_state_action_features(
                    state_list, action, player
                )

                # Store for later Q-value assignment
                training_samples.append(
                    {
                        "features": features,
                        "player": player,
                        "move_idx": move_idx,
                        "action": action,
                    }
                )

                # Make the move using board processor
                state_list[action].append(player)
                board = self.board_processor._build_board_matrix(state_list)
                self.board_processor.board_history.append(board)

                # Check for winner after this move
                winner = self.detect_winner_from_features(state_list)

                if winner is not None:
                    # Game ended - assign Q-values
                    if verbose:
                        print(f"Game ended at move {move_idx + 1}")
                        print(f"Winner: {winner}")
                        self.board_processor.display_board()

                    # Assign Monte Carlo returns
                    results = []
                    for sample in training_samples:
                        # Calculate return based on outcome and player
                        moves_from_end = len(training_samples) - sample["move_idx"] - 1

                        if winner == 0:  # Draw
                            q_value = 0.0
                        elif winner == sample["player"]:  # This player won
                            q_value = discount**moves_from_end
                        else:  # This player lost
                            q_value = -(discount**moves_from_end)

                        results.append(
                            (
                                sample["features"],
                                q_value,
                                {
                                    "player": sample["player"],
                                    "action": sample["action"],
                                    "move_idx": sample["move_idx"],
                                    "winner": winner,
                                    "code": code,
                                },
                            )
                        )

                    return results

                # Switch players
                player *= -1

            # Game didn't end (shouldn't happen with valid position codes)
            if verbose:
                print(f"Warning: Game didn't end naturally for code {code}")
            return []

        except Exception as e:
            print(f"Error processing code {code}: {e}")
            return []

    def process_position_codes_file(
        self, filepath: str, max_codes: Optional[int] = None, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a file containing position codes (one per line).

        Returns:
            (features_array, q_values_array)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            codes = [line.strip() for line in f if line.strip()]

        if max_codes:
            codes = codes[:max_codes]

        print(f"Processing {len(codes)} position codes...")

        all_features = []
        all_q_values = []

        for idx, code in enumerate(codes):
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(codes)} codes...")

            samples = self.process_position_code(code, verbose=(verbose and idx < 5))

            for features, q_value, metadata in samples:
                all_features.append(features)
                all_q_values.append(q_value)

        features_array = np.array(all_features)
        q_values_array = np.array(all_q_values)

        print(f"\nProcessed {len(codes)} games")
        print(f"Total training samples: {len(all_features)}")
        print(f"Features shape: {features_array.shape}")
        print(f"Q-value statistics:")
        print(f"  Mean: {q_values_array.mean():.3f}")
        print(f"  Std:  {q_values_array.std():.3f}")
        print(f"  Min:  {q_values_array.min():.3f}")
        print(f"  Max:  {q_values_array.max():.3f}")
        print(f"  Positive (wins): {(q_values_array > 0).sum()}")
        print(f"  Negative (losses): {(q_values_array < 0).sum()}")
        print(f"  Zero (draws): {(q_values_array == 0).sum()}")

        return features_array, q_values_array

    def save_training_data_for_bayesian(
        self, features: np.ndarray, q_values: np.ndarray, output_file: str
    ):
        """
        Save training data in format expected by bayesian_trainer.py

        Format: return_value \t placeholder \t placeholder \t placeholder \t placeholder \t feature1 \t feature2 ...
        """
        num_samples = features.shape[0]

        # Create placeholder columns
        placeholders = np.zeros((num_samples, 4))

        # Combine: q_values, placeholders, features
        data = np.column_stack([q_values.reshape(-1, 1), placeholders, features])

        # Save with header
        header = "return\tbackup\tterm\tval\tcode"
        for i in range(features.shape[1]):
            header += f"\tfeature_{i}"

        np.savetxt(output_file, data, delimiter="\t", header=header, comments="")
        print(f"\nSaved {num_samples} training examples to {output_file}")


def main():
    """Process Connect 4 position codes into training data."""
    import argparse

    parser = argparse.ArgumentParser(description="Process Connect 4 position codes")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with position codes (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/input/r139_0.txt",
        help="Output file for Bayesian trainer",
    )
    parser.add_argument(
        "--max-codes", type=int, default=None, help="Maximum number of codes to process"
    )
    parser.add_argument("--test", action="store_true", help="Run dimension tests")
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed processing info"
    )
    parser.add_argument("--test-code", type=str, help="Test a single position code")

    args = parser.parse_args()

    processor = PositionCodeProcessor()

    if args.test:
        processor.test_feature_dimensions()
        return

    if args.test_code:
        print(f"Testing position code: {args.test_code}")
        samples = processor.process_position_code(args.test_code, verbose=True)
        print(f"\nGenerated {len(samples)} training samples")
        for i, (features, q_value, metadata) in enumerate(samples[:5]):
            print(
                f"Sample {i}: Q={q_value:+.3f}, Player={metadata['player']}, Action={metadata['action']}"
            )
        return

    # Process position codes file
    features, q_values = processor.process_position_codes_file(
        args.input, max_codes=args.max_codes, verbose=args.verbose
    )

    # Save for Bayesian trainer
    processor.save_training_data_for_bayesian(features, q_values, args.output)


def tent(code):
    processor = PositionCodeProcessor()
    print(f"Testing position code: {code}")
    samples = processor.process_position_code(code, verbose=True)
    print(f"\nGenerated {len(samples)} training samples")
    for i, (features, q_value, metadata) in enumerate(samples):
        unique_feats, feat_counts = np.unique(
            features[features != 0], return_counts=True
        )
        unique_feats = unique_feats.astype(np.int8)
        print(
            f"Sample {i}: Q={q_value:+.3f}, Player={metadata['player']}, "
            f"Action={metadata['action']}, Features={dict(zip(unique_feats, feat_counts))}"
        )
    return


if __name__ == "__main__":
    tent("6YxAvYqb3XS")
