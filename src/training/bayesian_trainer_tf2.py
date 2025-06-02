"""
Improved Bayesian HMC trainer for deep neural networks.
TensorFlow 2.x version with eager execution and modern API.
"""
import sys
import time
import os
from typing import Tuple, Generator, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .model_utils import BayesianDNN, normalize_features

tfd = tfp.distributions


class TF2BayesianTrainer:
    """
    TensorFlow 2.x version of the Bayesian HMC trainer.
    Uses eager execution and modern TF2 APIs.
    """

    def __init__(
        self,
        run_index: str,
        leapfrog_steps: int,
        data_dir: str = "data",
        output_dir: str = "data/models",
    ) -> None:
        """
        Initialize the Bayesian trainer.

        Args:
            run_index: Identifier for this training run
            leapfrog_steps: Number of leapfrog steps for HMC
            data_dir: Directory containing input data
            output_dir: Directory for saving outputs
        """
        self.run_index = str(run_index)
        self.leapfrog_steps = int(leapfrog_steps)
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # File paths
        self.input_filename = os.path.join(
            data_dir, f"r139_{self.run_index}.txt"
        )
        self.output_pattern = os.path.join(
            output_dir, f"sampling_parameters_23_xr{self.run_index}_{{:03d}}.txt"
        )
        self.reference_pattern = os.path.join(output_dir, "reference.txt")
        self.fit_pattern = os.path.join(
            output_dir, f"fit_127_xr{self.run_index}_{{:03d}}.txt"
        )

        # Model hyperparameters
        self.master_dtype = tf.float32
        self.first_layer_size = 16
        self.second_layer_size = 16
        self.num_burnin_steps = 5000
        self.num_results = 2
        self.number_vars = 138
        self.window_length = 58368
        self.max_rows = 1000000
        self.number_days_ret_forward = 1

        # Training hyperparameters
        self.initial_prior_scale = 200
        self.studentT_scale = 100
        self.initial_step_size = 0.01
        self.likelihood_scale = 0.3

        # Initialize model
        self.bdnn_model = BayesianDNN(
            num_vars=self.number_vars,
            first_layer_size=self.first_layer_size,
            second_layer_size=self.second_layer_size,
        )
        self.num_model_parameters = self.bdnn_model.total_params

        # Data containers
        self.multifactor_data: Optional[np.ndarray] = None
        self.dates_data: Optional[pd.DataFrame] = None
        self.num_windows = 1

        # Training state variables
        self.p_mean = tf.Variable(
            tf.zeros(self.num_model_parameters, dtype=self.master_dtype),
            trainable=False,
            name="p_mean"
        )
        self.p_scale = tf.Variable(
            tf.fill([self.num_model_parameters], self.initial_prior_scale),
            trainable=False,
            name="p_scale"
        )
        self.step_size = tf.Variable(
            self.initial_step_size,
            trainable=False,
            name="step_size"
        )

        self._setup_tensorflow()

    def _setup_tensorflow(self) -> None:
        """Setup TensorFlow 2.x configuration."""
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

        # Set CPU parallelism
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)

        # Set seeds
        current_time = int(time.time())
        tf.random.set_seed(current_time)
        np.random.seed(current_time)

        print("TensorFlow 2.x configured")
        print(f"Model parameters: {self.num_model_parameters}")
        print(f"Eager execution: {tf.executing_eagerly()}")

    def load_data(self) -> None:
        """Load training data from file."""
        print(f"Loading data from: {self.input_filename}")

        if not os.path.exists(self.input_filename):
            raise FileNotFoundError(f"Input file not found: {self.input_filename}")

        # Load data columns (same as original)
        xr_position = range(3, 4)
        data_position = range(5, 5 + self.number_vars)
        used_cols = [i for j in (xr_position, data_position) for i in j]

        # Load main data
        self.multifactor_data = np.genfromtxt(
            self.input_filename,
            skip_header=1,
            usecols=used_cols,
            missing_values="NA",
            max_rows=self.max_rows,
            delimiter="\t",
        )

        # Load dates data
        self.dates_data = pd.read_csv(
            self.input_filename,
            header=0,
            usecols=list(range(5)),
            nrows=self.max_rows,
            sep="\t",
        )
        self.dates_data.columns = ["date", "backup", "term", "val", "code"]

        print(f"Data shape: {self.multifactor_data.shape}")

        # Process dates (same as original)
        self.num_windows = 1  # Same as original
        print(f"Processing {self.num_windows} windows")

    def load_initial_parameters(self) -> np.ndarray:
        """Load initial parameters from previous run or create zeros."""
        try:
            training_params = np.genfromtxt(
                self.output_pattern.format(0), max_rows=2, delimiter="\t"
            )
            print("Loaded initial parameters from previous run")
            return training_params[0, :].astype(self.master_dtype.as_numpy_dtype)
        except Exception as e:
            print(f"No initial parameters found ({e}), using zeros")
            return np.zeros(
                self.num_model_parameters, dtype=self.master_dtype.as_numpy_dtype
            )

    def generate_sample_data(self) -> Generator[np.ndarray, None, None]:
        """Data generator for training windows."""
        if self.multifactor_data is None or self.dates_data is None:
            raise RuntimeError("Data must be loaded before generating samples")

        list1 = self.dates_data["date"].values
        lt = np.sort(np.unique(list1))
        max_pop = self.window_length
        date_of_interest = 1
        test_set = list1 == lt[date_of_interest]

        for counter in range(self.num_windows):
            test_population = np.sum(test_set)
            trim = test_population - max_pop

            if trim >= 0:
                update_set = np.where(test_set)[0][trim:]
                print("No bootstrapping needed")
            else:
                update_set_incomplete = np.where(test_set)[0]
                indices = (np.random.rand(max_pop) * len(update_set_incomplete)).astype(
                    int
                )
                update_set = update_set_incomplete[indices]
                print("Bootstrapping applied")

            train_set = update_set
            date_of_interest += 1
            test_set = list1 == lt[date_of_interest]

            print(f"Test date: {lt[date_of_interest]}")
            qqq = np.unique(list1[update_set], return_counts=True)
            print(
                f"Train date from: {qqq[0][0]} ({qqq[1][0]}) to: {qqq[0][-1]} ({qqq[1][-1]})"
            )

            sub_factor_ = self.multifactor_data[train_set, 1:]
            ret_ = np.expand_dims(
                np.sum(
                    self.multifactor_data[train_set, : self.number_days_ret_forward], 1
                ),
                1,
            )
            c = np.concatenate((ret_, sub_factor_), axis=1)

            # Write reference (same as original)
            date_identifier = (
                str(lt[date_of_interest]) + "\t" + format(counter, "03d") + "\n"
            )
            with open(self.reference_pattern, "a") as ref_file:
                ref_file.write(date_identifier)

            yield c.astype(self.master_dtype.as_numpy_dtype)

    @tf.function
    def model_log_prob(
        self,
        ret: tf.Tensor,
        feat: tf.Tensor,
        p: tf.Tensor,
    ) -> tf.Tensor:
        """
        Model log probability computation.

        Args:
            ret: Returns tensor
            feat: Features tensor
            p: Current parameters

        Returns:
            Log probability tensor
        """
        # Prior probability
        rv_p = tfd.Independent(
            tfd.Normal(loc=self.p_mean, scale=self.p_scale),
            reinterpreted_batch_ndims=1
        )

        # Normalize features
        normalized_feat = tf.py_function(
            lambda x: normalize_features(x.numpy(), method="power").astype(
                self.master_dtype.as_numpy_dtype
            ),
            [feat],
            self.master_dtype,
        )
        normalized_feat.set_shape(feat.shape)

        # Get model predictions
        mz = self.bdnn_model(normalized_feat, p)

        # Likelihood
        rv_observed = tfd.Normal(loc=mz, scale=self.likelihood_scale)

        return rv_p.log_prob(p) + tf.reduce_sum(rv_observed.log_prob(ret))

    def run_hmc_sampling(self, batch_data: tf.Tensor, initial_state: tf.Tensor) -> Tuple:
        """
        Run HMC sampling for one window.

        Args:
            batch_data: Input data for this window
            initial_state: Initial parameter state

        Returns:
            Tuple of (samples, kernel_results, final_state)
        """
        # Split data
        observed_return = batch_data[:, 0:1]
        observed_features = batch_data[:, 1:]

        # Define target log probability function
        def target_log_prob_fn(params):
            return self.model_log_prob(observed_return, observed_features, params)

        # Create HMC kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=self.leapfrog_steps,
            step_size=self.step_size,
        )

        # Add step size adaptation
        adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=self.num_burnin_steps,
            target_accept_prob=0.6,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )

        # Sample from chain
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            num_burnin_steps=self.num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr,
        )

        return samples, kernel_results, batch_data

    def update_training_state(self, samples: tf.Tensor, kernel_results) -> None:
        """Update training state variables after sampling."""
        # Update parameter statistics
        new_p_mean, new_p_var = tf.nn.moments(samples, axes=[0])
        new_p_scale = tf.math.maximum(tf.sqrt(new_p_var), 100.0)

        # Update step size from adaptation
        new_step_size = tf.math.maximum(
            0.0001,
            kernel_results.new_step_size if hasattr(kernel_results, 'new_step_size')
            else kernel_results.inner_results.step_size
        )

        # Assign new values
        self.p_mean.assign(new_p_mean)
        self.p_scale.assign(new_p_scale)
        self.step_size.assign(new_step_size)

    def train(self) -> None:
        """Run the complete training process."""
        print("\nBayesian BDNN Training (TensorFlow 2.x Version)")
        print("=" * 50)
        print(f"Run Index: {self.run_index}")
        print(f"Leapfrog Steps: {self.leapfrog_steps}")
        print(f"Model Parameters: {self.num_model_parameters}")
        print(f"Burn-in Steps: {self.num_burnin_steps}")
        print(f"Results per window: {self.num_results}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

        # Load data
        self.load_data()

        # Load initial parameters
        initial_params = self.load_initial_parameters()
        current_state = tf.Variable(initial_params, dtype=self.master_dtype)

        # Training loop
        global_start_time = time.time()
        data_generator = self.generate_sample_data()

        for s in range(self.num_windows):
            print(f"\nTraining Window {s + 1}/{self.num_windows}")
            print("-" * 30)

            # Get data for this window
            batch_data = next(data_generator)
            batch_data_tensor = tf.constant(batch_data, dtype=self.master_dtype)

            print("Starting HMC sampling...")
            start_time = time.time()

            # Run HMC sampling
            try:
                samples, kernel_results, batch_data_final = self.run_hmc_sampling(
                    batch_data_tensor, current_state
                )

                # Update training state
                self.update_training_state(samples, kernel_results)

                # Update current state for next window
                current_state.assign(self.p_mean)

            except Exception as e:
                print(f"Error during sampling: {e}")
                raise

            end_time = time.time()

            # Print results
            print(f"✓ Completed in {end_time - start_time:.2f} seconds")
            print(f"Final step size: {self.step_size.numpy():.6f}")
            print(f"Mean of samples (first 4): {tf.reduce_mean(samples[:, 0:4], axis=0).numpy()}")
            print(f"Parameter means (first 4): {self.p_mean[0:4].numpy()}")
            print(f"Parameter scales (first 4): {self.p_scale[0:4].numpy()}")

            # Generate fit data
            normalized_features = normalize_features(batch_data[:, 1:], method="power")
            fit_data = self.bdnn_model(
                tf.constant(normalized_features, dtype=self.master_dtype),
                samples[-1, :]
            )
            fit_data_complete = np.concatenate((batch_data, fit_data.numpy()), axis=1)

            # Save results
            print("Saving results...")
            np.savetxt(
                self.output_pattern.format(s),
                samples.numpy(),
                delimiter="\t",
                fmt="%.3g"
            )
            np.savetxt(
                self.fit_pattern.format(s),
                fit_data_complete,
                delimiter="\t",
                fmt="%.5g",
            )
            print(f"✓ Saved to {self.output_pattern.format(s)}")

        global_end_time = time.time()
        print("\n🎉 Training completed successfully!")
        print(f"Total time: {global_end_time - global_start_time:.2f} seconds")
        print(f"Output files saved to: {self.output_dir}")


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 3:
        print(
            "Usage: python bayesian_trainer_tf2.py <run_index> <leapfrog_steps> [data_dir] [output_dir]"
        )
        sys.exit(1)

    run_index = sys.argv[1]
    leapfrog_steps = int(sys.argv[2])
    data_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "data/models"

    print("Bayesian HMC Training (TensorFlow 2.x)")
    print("=" * 40)
    print(f"Run Index: {run_index}")
    print(f"Leapfrog Steps: {leapfrog_steps}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")

    trainer = TF2BayesianTrainer(run_index, leapfrog_steps, data_dir, output_dir)
    trainer.train()


if __name__ == "__main__":
    main()