""" wasserstein_metric.py - This file contains the WassersteinOne class, which
is a omnifold classifier performance metric. It calculates the 1D wasserstein
metric between two distributions over the dimensions used in plotting, and then sums
them to get a single performance metric.

It is a subclass of the BaseMetric class, which is a subclass of torchmetrics.Metric.

Author: Kevin Greif
Last updated 05/08/2024
python3
"""


import numpy as np
import scipy
from pytorch_lightning.utilities.rank_zero import *

from base_metric import BaseMetric
import utils.plotting_utils as pu


class WassersteinOne(BaseMetric):
    """ A BaseMetric subclass which implements the 1D wasserstein 
    based performance metrics for omnifold classifiers.
    """

    # Can keep the same __init__ and update functions as the base class

    def compute(self, from_torch=True, **kwargs):
        """ compute - Actually compute the sum of the wasserstein distances
        over the dimensions used in plotting.

        This calls the parent classes compute function

        Arguments:
        from_torch - Optional, default True. If True, will convert the state tensors
            to numpy. If false assumes state tensors are alreeady numpy
        Returns: the sum of the wasserstein distances, and a dictionary of plots
        with form {plot_name: stored location of .png file}. If draw_plots is False,
        then this dictionary is empty.
        """

        # Call parent compute function
        plot_dict = super().compute(from_torch=from_torch, **kwargs)

        # Separate weights to use in wasserstein calculation
        source_weight = self.end_weights[self.target == 0]
        target_weight = self.start_weights[self.target == 1]

        # Results list
        results = []

        # Loop through plotting dimensions
        for i, (key, hist_dict) in enumerate(self.hist_info.items()):

            # If we are not using this dimension for W1 computation, continue
            if not hist_dict['w1_eval']:
                continue

            # Slice this dimension
            this_dim = self.plotting[:,i]

            # Separate into source / target
            source_dist = this_dim[self.target == 0]
            target_dist = this_dim[self.target == 1]

            # Calculate 1D wasserstein distance
            this_wass = scipy.stats.wasserstein_distance(source_dist, target_dist, u_weights=source_weight, v_weights=target_weight)

            # Append to results list
            results.append(this_wass)

        # Return the sum over all plotting dimensions, and the plot dictionary
        return np.sum(np.array(results)), plot_dict