""" wasserstein_metric.py - This file contains the WassersteinOne class, which
is a omnifold classifier performance metric. It calculates the 1D wasserstein
metric between two distributions over the dimensions used in plotting, and then sums
them to get a single performance metric.

It is a subclass of torchmetrics.Metric, which lets us neatly handle calculation
of the metric across devices in distributed training.

Author: Kevin Greif
Last updated 03/24/2025
python3
"""

import numpy as np
import scipy
import torchmetrics


class WassersteinOne(torchmetrics.Metric):
    """A BaseMetric subclass which implements the 1D wasserstein
    based performance metrics for omnifold classifiers.
    """

    def __init__(self, **kwargs):
        """__init__ - Init function for the class. Defines the state of the metric,
        and reads yaml config file to get the dimensions to use for calculating
        the wasserstein distances.

        No arguments or returns
        """
        super().__init__(**kwargs)
        self.add_state("observables", default=[], dist_reduce_fx="cat")
        self.add_state("weights", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, observables, weights, labels):
        """update - Function to update the metric state on each batch."""
        self.observables.append(observables)
        self.weights.append(weights)
        self.target.append(labels)

    def compute(self, from_torch=True):
        """compute - Compute the sum of the wasserstein distances
        over the dimensions used in plotting.

        This calls the parent classes compute function

        Arguments:
        from_torch - Optional, default True. If True, will convert the state tensors
            to numpy. If false assumes state tensors are already numpy

        Returns:

        """

        # Convert to numpy if needed
        if from_torch:

            # Concatenate list states
            observables = torchmetrics.utilities.dim_zero_cat(self.observables)
            weights = torchmetrics.utilities.dim_zero_cat(self.weights)
            target = torchmetrics.utilities.dim_zero_cat(self.target)

            observables = observables.cpu().numpy()
            weights = weights.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()

        # Else the state already contains numpy, just need to concatenate
        else:
            observables = np.concatenate(self.observables, axis=0)
            weights = np.concatenate(self.weights, axis=0).flatten()
            target = np.concatenate(self.target, axis=0).flatten()

        # Mask any NaN, inf, or negative values in the weights
        mask = ~np.isnan(weights) & ~np.isinf(weights) & (weights >= 0)
        observables = observables[mask]
        weights = weights[mask]
        target = target[mask]

        # Separate weights to use in wasserstein calculation
        source_weight = weights[target == 0]
        target_weight = weights[target == 1]

        # Results list
        results = []

        # Loop through observables
        for i in range(observables.shape[1]):

            # Slice this dimension
            this_dim = observables[:, i]

            # Separate into source / target
            source_dist = this_dim[target == 0]
            target_dist = this_dim[target == 1]

            # Calculate 1D wasserstein distance
            this_wass = scipy.stats.wasserstein_distance(
                source_dist,
                target_dist,
                u_weights=source_weight,
                v_weights=target_weight,
            )

            # Append to results list
            results.append(this_wass)

        # Return the sum over all plotting dimensions
        return np.sum(np.array(results))

    def reset(self):
        """reset - Reset the metric state."""
        self.observables = []
        self.weights = []
        self.target = []

        super().reset()
