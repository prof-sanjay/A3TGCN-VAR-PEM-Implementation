import random

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed used by Python, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)