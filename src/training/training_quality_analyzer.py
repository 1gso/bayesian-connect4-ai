#!/usr/bin/env python3
"""
Bayesian Training Quality Analyzer
Analyzes fit files to assess training quality through correlation analysis and XY plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


class BayesianTrainingAnalyzer:
    def __init__(self, models_dir="data/models"):
        self.models_dir = Path(models_dir)
        self.fit_files = []
        self.sampling_files = []

    def find_training_files(self, run_index="8"):
        """Find all fit and sampling files for a given run."""
        fit_pattern = f"fit_*_xr{run_index}_*.txt"
        sampling_pattern = f"sampling_parameters_*_xr{run_index}_*.txt"

        self.fit_files = list(self.models_dir.glob(fit_pattern))
        self.sampling_files = list(self.models_dir.glob(sampling_pattern))

        print(f"Found {len(self.fit_files)} fit files")
        print(f"Found {len(self.sampling_files)} sampling files")

        return self.fit_files, self.sampling_files

    def load_fit_data(self, fit_file):
        """Load fit data and separate original vs predictions."""
        try:
            data = np.loadtxt(fit_file, delimiter="\t")
            print(f"Loaded fit data shape: {data.shape}")

            # Based on your code: [original_data | predictions]
            # Original data: returns (col 0) + features (cols 1:139)
            # Predictions: last column

            original_returns = data[:, 0]  # Target variable
            original_features = data[:, 1:139]  # Features
            predictions = data[:, -1]  # Model predictions

            return {
                "original_returns": original_returns,
                "original_features": original_features,
                "predictions": predictions,
                "raw_data": data,
            }
        except Exception as e:
            print(f"Error loading {fit_file}: {e}")
            return None

    def analyze_prediction_quality(self, fit_data):
        """Analyze prediction vs actual correlation."""
        actual = fit_data["original_returns"]
        predicted = fit_data["predictions"]

        # Calculate correlation
        correlation = np.corrcoef(actual, predicted)[0, 1]

        # Calculate R-squared
        r_squared = correlation**2

        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))

        # Statistical significance
        stat_result = stats.pearsonr(actual, predicted)

        metrics = {
            "correlation": correlation,
            "r_squared": r_squared,
            "rmse": rmse,
            "mae": mae,
            "p_value": stat_result.pvalue,
            "n_samples": len(actual),
        }

        return metrics, actual, predicted

    def create_xy_plots(self, actual, predicted, metrics, save_path=None):
        """Create XY scatter plot of predictions vs actual."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        ax1.scatter(actual, predicted, alpha=0.6, s=1)
        ax1.plot(
            [actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2
        )
        ax1.set_xlabel("Actual Returns")
        ax1.set_ylabel("Predicted Returns")
        ax1.set_title(f'Predictions vs Actual\nR² = {metrics["r_squared"]:.4f}')
        ax1.grid(True, alpha=0.3)

        # Add correlation info
        textstr = f"""Correlation: {metrics['correlation']:.4f}
R²: {metrics['r_squared']:.4f}
RMSE: {metrics['rmse']:.6f}
MAE: {metrics['mae']:.6f}
p-value: {metrics['p_value']:.2e}
N: {metrics['n_samples']:,}"""

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax1.text(
            0.05,
            0.95,
            textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        # Residuals plot
        residuals = actual - predicted
        ax2.scatter(predicted, residuals, alpha=0.6, s=1)
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("Predicted Returns")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        return fig

    def analyze_parameter_convergence(self, sampling_files):
        """Analyze parameter convergence from sampling files."""
        if not sampling_files:
            print("No sampling files found")
            return None

        convergence_data = []

        for file_path in sampling_files:
            try:
                samples = np.loadtxt(file_path, delimiter="\t")
                print(f"Loaded sampling data shape: {samples.shape}")

                # Calculate some convergence metrics
                if samples.shape[0] > 1:
                    # Parameter means across samples
                    param_means = np.mean(samples, axis=0)
                    param_stds = np.std(samples, axis=0)

                    convergence_data.append(
                        {
                            "file": file_path.name,
                            "n_samples": samples.shape[0],
                            "n_params": samples.shape[1],
                            "mean_param_std": np.mean(param_stds),
                            "max_param_std": np.max(param_stds),
                            "param_range": np.ptp(samples, axis=0).mean(),
                        }
                    )

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return pd.DataFrame(convergence_data)

    def full_analysis(self, run_index="8"):
        """Run complete training quality analysis."""
        print("🔍 Bayesian Training Quality Analysis")
        print("=" * 50)

        # Find files
        fit_files, sampling_files = self.find_training_files(run_index)

        if not fit_files:
            print(f"❌ No fit files found for run index '{run_index}'")
            return

        # Analyze each fit file
        results = []

        for i, fit_file in enumerate(fit_files):
            print(f"\n📊 Analyzing {fit_file.name}")
            print("-" * 30)

            fit_data = self.load_fit_data(fit_file)
            if fit_data is None:
                continue

            # Analyze prediction quality
            metrics, actual, predicted = self.analyze_prediction_quality(fit_data)
            results.append(metrics)

            # Print metrics
            print(f"Correlation: {metrics['correlation']:.4f}")
            print(f"R-squared: {metrics['r_squared']:.4f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"P-value: {metrics['p_value']:.2e}")
            print(f"Samples: {metrics['n_samples']:,}")

            # Create plots
            plot_path = self.models_dir / f"quality_analysis_{fit_file.stem}.png"
            fig = self.create_xy_plots(actual, predicted, metrics, plot_path)
            plt.show()

        # Analyze parameter convergence
        if sampling_files:
            print(f"\n🎯 Parameter Convergence Analysis")
            print("-" * 30)
            conv_data = self.analyze_parameter_convergence(sampling_files)
            if conv_data is not None:
                print(conv_data)

        # Summary
        if results:
            print(f"\n📈 Summary Statistics")
            print("-" * 30)
            df_results = pd.DataFrame(results)
            print(f"Mean Correlation: {df_results['correlation'].mean():.4f}")
            print(f"Mean R²: {df_results['r_squared'].mean():.4f}")
            print(f"Mean RMSE: {df_results['rmse'].mean():.6f}")

            # Training quality assessment
            avg_r2 = df_results["r_squared"].mean()
            if avg_r2 > 0.7:
                print("✅ Excellent training quality!")
            elif avg_r2 > 0.5:
                print("✅ Good training quality")
            elif avg_r2 > 0.3:
                print("⚠️ Moderate training quality")
            else:
                print("❌ Poor training quality - consider adjusting model")

        return results


def main():
    """Main analysis function."""
    analyzer = BayesianTrainingAnalyzer()

    # Run analysis for your current training (run index "8")
    results = analyzer.full_analysis(run_index="8")

    return results


if __name__ == "__main__":
    main()
