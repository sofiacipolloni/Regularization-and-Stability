from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    """Simple container for a dataset already split into train/test."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        """Number of training examples."""
        return self.X_train.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (columns of X)."""
        return self.X_train.shape[1]



# REAL DATASET

def load_real_dataset(test_size: float = 0.25, random_state: int = 0) -> Dataset:
    """
    Load the real-world dataset: sklearn's 'diabetes' dataset (regression,
    442 patients, 10 clinical features, target = a quantitative measure of
    disease progression one year after baseline).

    It is a small/medium-sized dataset, well suited to running many
    experiments quickly, and a genuine regression problem rather than a toy
    example.
    """
    data = load_diabetes()
    X, y = data.data, data.target  # pylint: disable=no-member  (false positive)

    # random_state is the seed: fixing it makes the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardization of the features (mean 0, std 1). The scaler is fit on the
    # train set only (otherwise: data leakage) and then reused on the test set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Center the target using the train mean only (data leakage)
    y_mean = y_train.mean()
    y_train = y_train - y_mean
    y_test = y_test - y_mean

    return Dataset(
        name="real_diabetes",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )



# SYNTHETIC DATASET

def generate_synthetic_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    noise_std: float = 1.0,
    test_size: float = 0.25,
    random_state: int = 0,
) -> Dataset:
    """
    Generate a synthetic linear regression dataset.

    Generative process (deterministic given the seed):
      1. X: n_samples points, each with n_features coordinates i.i.d. from
         a standard Normal(0, 1).
      2. w_true: a fixed 'ground truth' weight vector, with about 30% of
         its entries set to zero (so the problem also has a sparsity
         structure, useful later when comparing Ridge and Lasso).
      3. y = X @ w_true + Gaussian noise with standard deviation noise_std.
    """
   
    rng = np.random.default_rng(random_state)   # fixed seed

    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)) # gaussian features: already mean 0 / variance 1 --> no scaling 

   
    w_true = rng.normal(loc=0.0, scale=1.0, size=n_features) # true weights to generate y. 30% set to 0 --> no effect on y
    n_zero = int(round(0.3 * n_features))
    zero_idx = rng.choice(n_features, size=n_zero, replace=False)
    w_true[zero_idx] = 0.0

    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples) # inear signal + gaussian noise
    y = X @ w_true + noise


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return Dataset(
        name="synthetic",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )