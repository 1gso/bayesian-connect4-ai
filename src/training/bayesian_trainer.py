"""
Improved Bayesian HMC trainer for deep neural networks.
CPU/GPU version converted from original IPU code with better organization.
"""
import sys
import time
import os
# from multiprocessing.util import sub_debug
from typing import Tuple, Generator, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

# from concurrent.futures import ProcessPoolExecutor
# import multiprocessing as mp

from .model_utils import BayesianDNN, normalize_features

tfd = tfp.distributions


class CPUGPUBayesianTrainer:
    """
    Improved CPU/GPU version of the Bayesian HMC trainer.
    Optimized for Azure ND40s_v3 (40 cores, 661GB RAM) or similar systems.
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
            data_dir,  f"r139_{self.run_index}.txt"
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

        self._setup_tensorflow()

    def _setup_tensorflow(self) -> None:
        """Setup TensorFlow for CPU/GPU (no IPU)."""
        # Suppress warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # TensorFlow 1.x configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.inter_op_parallelism_threads = 0  # Use all available cores
        config.intra_op_parallelism_threads = 0  # Use all available cores

        self.sess = tf.InteractiveSession(config=config)

        # Set seeds (TF 1.x style)
        current_time = int(time.time())
        tf.set_random_seed(current_time)
        np.random.seed(current_time)

        print("TensorFlow configured for CPU/GPU")
        print(f"Model parameters: {self.num_model_parameters}")

        # TensorFlow 1.x device listing
        try:
            from tensorflow.python.client import device_lib

            devices = device_lib.list_local_devices()
            device_names = [d.name for d in devices]
            print(f"Available devices: {device_names}")
        except Exception as e:
            print(f"Device listing not available: {e}")

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
        # l1 = self.dates_data["date"].values
        # l2 = np.sort(np.unique(l1))
        # start_position = np.where(l2 == l1[self.window_length])[0][0]
        # end_position = l2.shape[0]
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

    def generate_sample_mini(self) -> Generator[np.ndarray, None, None]:
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

    def model_log_prob(
        self,
        ret: tf.Tensor,
        feat: tf.Tensor,
        p_mean: tf.Tensor,
        p_scale: tf.Tensor,
        p: tf.Tensor,
    ) -> tf.Tensor:
        """
        Model log probability computation.

        Args:
            ret: Returns tensor
            feat: Features tensor
            p_mean: Parameter means
            p_scale: Parameter scales
            p: Current parameters

        Returns:
            Log probability tensor
        """
        # Prior probability
        rv_p = tfd.Independent(
            tfd.Normal(loc=p_mean, scale=p_scale), reinterpreted_batch_ndims=1
        )

        # Normalize features and get predictions
        normalized_feat = tf.py_function(
            lambda x: normalize_features(x.numpy(), method="power").astype(
                self.master_dtype.as_numpy_dtype
            ),
            [feat],
            self.master_dtype,
        )
        normalized_feat.set_shape(feat.shape)

        # Get model predictions using improved BDNN
        mz = self.bdnn_model(normalized_feat, p)

        # Likelihood
        rv_observed = tfd.Normal(loc=mz, scale=self.likelihood_scale)

        return rv_p.log_prob(p) + tf.reduce_sum(rv_observed.log_prob(ret))

    def build_hmc_graph(self) -> Tuple[callable, tf.data.Iterator]:
        """Build HMC computation graph for CPU/GPU."""
        # Create dataset (TF 1.x style)
        dataset = tf.data.Dataset.from_generator(
            self.generate_sample_mini,
            output_types=self.master_dtype,
            output_shapes=[self.window_length, 1 + self.number_vars],
        )
        iterator = dataset.make_initializable_iterator()

        # Load initial parameters
        initial_params = self.load_initial_parameters()

        # Get next window data
        batch_data = iterator.get_next()
        observed_return, observed_features = tf.split(
            batch_data, num_or_size_splits=[1, self.number_vars], axis=1
        )

        # Initialize variables (TF 1.x style)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            p_mean = tf.get_variable(
                "p_mean",
                shape=[self.num_model_parameters],
                initializer=tf.zeros_initializer(dtype=self.master_dtype),
                dtype=self.master_dtype,
                trainable=False,
            )

            p_scale = tf.get_variable(
                "p_scale",
                shape=[self.num_model_parameters],
                initializer=tf.constant_initializer(
                    self.initial_prior_scale, dtype=self.master_dtype
                ),
                dtype=self.master_dtype,
                trainable=False,
            )

            initial_chain_state = tf.get_variable(
                "initial_chain_state",
                shape=[self.num_model_parameters],
                initializer=tf.constant_initializer(
                    initial_params, dtype=self.master_dtype
                ),
                dtype=self.master_dtype,
                trainable=False,
            )

            step_size = tf.get_variable(
                name="step_size",
                initializer=tf.constant(
                    self.initial_step_size, dtype=self.master_dtype
                ),
                trainable=False,
                use_resource=True,
            )

        def hmc_graph():
            """HMC computation graph."""

            # Target log probability function
            def target_log_prob_fn(*args):
                return self.model_log_prob(
                    observed_return, observed_features, p_mean, p_scale, *args
                )

            # HMC kernel
            hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    num_leapfrog_steps=self.leapfrog_steps,
                    step_size=step_size,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                        target_rate=0.6,
                        num_adaptation_steps=self.num_burnin_steps,
                        decrement_multiplier=0.1,
                    ),
                    state_gradients_are_stopped=False,
                ),
                bijector=[tfp.bijectors.Identity()],
            )

            # Sample from chain
            p, kernel_results = tfp.mcmc.sample_chain(
                num_results=self.num_results,
                num_burnin_steps=self.num_burnin_steps,
                current_state=initial_chain_state,
                kernel=hmc_kernel,
            )

            # Update parameters
            new_p_mean, new_p_scale = tf.nn.moments(p, axes=[0])
            new_p_scale = tf.math.maximum(new_p_scale, 100)
            new_step_size = tf.math.maximum(
                0.0001, kernel_results.inner_results.extra.step_size_assign[-1]
            )

            # Assign updates (TF 1.x style)
            p_mean_upd = tf.assign(p_mean, new_p_mean)
            p_scale_upd = tf.assign(p_scale, new_p_scale)
            initial_chain_state_upd = tf.assign(initial_chain_state, new_p_mean)
            step_size_upd = tf.assign(step_size, new_step_size)

            with tf.control_dependencies(
                [p_mean_upd, p_scale_upd, initial_chain_state_upd, step_size_upd]
            ):
                p = tf.identity(p)

            return (
                p,
                kernel_results,
                p_mean,
                p_scale,
                step_size,
                initial_chain_state,
                batch_data,
            )

        return hmc_graph(), iterator

    def train(self) -> None:
        """Run the complete training process."""
        print("\nBayesian BDNN Training (CPU/GPU Version)")
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

        # Build graph
        hmc_ops, iterator = self.build_hmc_graph()

        # Initialize (TF 1.x style)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(iterator.initializer)

        # Training loop
        global_start_time = time.time()

        for s in range(self.num_windows):
            print(f"\nTraining Window {s + 1}/{self.num_windows}")
            print("-" * 30)
            print("Starting HMC sampling...")

            start_time = time.time()

            # Run HMC sampling
            try:
                results = self.sess.run(hmc_ops)
                (
                    samples_,
                    kernel_results_,
                    p_mean_before_,
                    p_scale_before_,
                    step_size_before_,
                    initial_chain_state_before_,
                    batch_data_,
                ) = results
            except Exception as e:
                print(f"Error during sampling: {e}")
                raise

            end_time = time.time()

            # Print results
            print(f"✓ Completed in {end_time - start_time:.2f} seconds")
            print(
                f"Final step size: {kernel_results_.inner_results.extra.step_size_assign[-1]:.6f}"
            )
            print(f"Mean of samples (first 4): {np.mean(samples_[:, 0:4], axis=0)}")
            print(f"Parameter means (first 4): {p_mean_before_[0:4]}")
            print(f"Parameter scales (first 4): {p_scale_before_[0:4]}")
            print(f"Step size: {step_size_before_:.6f}")

            # Generate fit data using the improved BDNN
            normalized_features = normalize_features(batch_data_[:, 1:], method="power")
            fit_data_ = self.sess.run(
                self.bdnn_model(
                    tf.constant(normalized_features, dtype=self.master_dtype),
                    tf.constant(samples_[-1, :], dtype=self.master_dtype),
                )
            )
            fit_data_complete_ = np.concatenate((batch_data_, fit_data_), axis=1)

            # Save results
            print("Saving results...")
            np.savetxt(
                self.output_pattern.format(s), samples_, delimiter="\t", fmt="%.3g"
            )
            np.savetxt(
                self.fit_pattern.format(s),
                fit_data_complete_,
                delimiter="\t",
                fmt="%.5g",
            )
            print(f"✓ Saved to {self.output_pattern.format(s)}")

        global_end_time = time.time()
        print("\n🎉 Training completed successfully!")
        print(f"Total time: {global_end_time - global_start_time:.2f} seconds")
        print(f"Output files saved to: {self.output_dir}")

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "sess"):
            self.sess.close()


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 3:
        print(
            "Usage: python bayesian_trainer.py <run_index> <leapfrog_steps> [data_dir] [output_dir]"
        )
        sys.exit(1)

    run_index = sys.argv[1]
    leapfrog_steps = int(sys.argv[2])
    data_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "data/models"

    print("Bayesian HMC Training")
    print("=" * 40)
    print(f"Run Index: {run_index}")
    print(f"Leapfrog Steps: {leapfrog_steps}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")

    trainer = CPUGPUBayesianTrainer(run_index, leapfrog_steps, data_dir, output_dir)

    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
