""" wasserstein_metric.py - This file contains the WassersteinOne class, which
is a omnifold classifier performance metric. It calculates the 1D wasserstein
metric between two distributions over the dimensions used in plotting, and then sums
them to get a single performance metric.

It is implemented in the TorchMetrics framework for easy integration with 
pytorch lightning.

Author: Kevin Greif
Last updated 02/29/2024
python3
"""


import torchmetrics
import numpy as np
import scipy

import plotting_utils as pu


class WassersteinOne(torchmetrics.Metric):
    """ A torchmetrics metric subclass which implements the 1D wasserstein 
    based performance metrics for omnifold classifiers.
    """

    def __init__(self, draw_plots=False, save_location=None, **kwargs):
        """ __init__ - Init function for the class. Both definees the state of the
        metric, and accepts some keyword arguments to control plot drawing if enabled.

        Arguments:
        draw_plots - Optional, default False. If True, will draw plots of the plot dimensions
        save_location - Optional, default None. If provided, will save the plots to this location

        Returns:
        None
        """
        super().__init__(**kwargs)
        self.add_state("plotting", default=[], dist_reduce_fx="cat")
        self.add_state("start_weights", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        self.draw_plots = draw_plots
        self.save_location = save_location


    def update(self, plotting, start_weights, preds, target):
        """ update - Function to update the metric state on each batch."""
        self.plotting.append(plotting)
        self.start_weights.append(start_weights)
        self.preds.append(preds)
        self.target.append(target)


    def compute(self):
        """ compute - Actually compute the sum of the wasserstein distances
        over the dimensions used in plotting.
        """

        # Concatenate list states
        plotting = torchmetrics.utilities.dim_zero_cat(self.plotting)
        start_weights = torchmetrics.utilities.dim_zero_cat(self.start_weights)
        preds = torchmetrics.utilities.dim_zero_cat(self.preds)
        target = torchmetrics.utilities.dim_zero_cat(self.target)

        # To numpy
        plotting = plotting.cpu().detach().numpy()
        start_weights = start_weights.cpu().detach().numpy().flatten()
        preds = preds.cpu().detach().numpy().flatten()
        target = target.cpu().detach().numpy().flatten()

        # Calculate weights
        probs = 1 / (1 + np.exp(-preds))
        derived_weights = probs / (1 - probs)
        mc_weight = derived_weights[target == 0]

        # If we want to draw plots, do so here
        if self.draw_plots:
            plot_dict = pu.make_logged_plots(
                plotting, target, start_weights, preds, save_location=self.save_location
            )

        # Results list
        results = np.zeros(plotting.shape[1])

        # Loop through plotting dimensions
        for i in range(plotting.shape[1]):

            # Slice this dimension
            this_dim = plotting[:,i]

            # Separate into data / MC
            mc_dist = this_dim[target == 0]
            data_dist = this_dim[target == 1]

            # Calculate 1D wasserstein distance
            this_wass = scipy.stats.wasserstein_distance(mc_dist, data_dist, u_weights=mc_weight)

            # Append to results list
            results[i] = this_wass

        # Return the sum over all plotting dimensions, and the plot dictionary if we made plots
        if self.draw_plots:
            return np.sum(results), plot_dict
        else:
            return np.sum(results)