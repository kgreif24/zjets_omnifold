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
from pytorch_lightning.utilities.rank_zero import *

import plotting_utils as pu


class WassersteinOne(torchmetrics.Metric):
    """ A torchmetrics metric subclass which implements the 1D wasserstein 
    based performance metrics for omnifold classifiers.
    """

    def __init__(self, hist_info, draw_plots=False, save_location=None, from_outputs=True, **kwargs):
        """ __init__ - Init function for the class. Both definees the state of the
        metric, and accepts some keyword arguments to control plot drawing if enabled.

        Arguments:
        hist_info - Dictionary describing the histograms used in the plotting
        draw_plots - Optional, default False. If True, will draw plots of the plot dimensions
        save_location - Optional, default None. If provided, will save the plots to this location

        Returns:
        None
        """
        super().__init__(**kwargs)
        self.add_state("plotting", default=[], dist_reduce_fx="cat")
        self.add_state("start_weights", default=[], dist_reduce_fx="cat")
        self.add_state("end_weights", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        self.hist_info = hist_info
        self.draw_plots = draw_plots
        self.save_location = save_location


    def update(self, plotting, start_weights, end_weights, labels):
        """ update - Function to update the metric state on each batch."""
        self.plotting.append(plotting)
        self.start_weights.append(start_weights)
        self.end_weights.append(end_weights)
        self.target.append(labels)


    def compute(self, from_torch=True, **kwargs):
        """ compute - Actually compute the sum of the wasserstein distances
        over the dimensions used in plotting.

        Arguments:
        from_torch - Optional, default True. If True, will convert the state tensors
            to numpy. If false assumes state tensors are alreeady numpy
        Returns: the sum of the wasserstein distances, and a dictionary of plots
        with form {plot_name: stored location of .png file}. If draw_plots is False,
        then this dictionary is empty.
        """

        # If from_torch, convert to numpy
        if from_torch:

            # Concatenate list states
            plotting = torchmetrics.utilities.dim_zero_cat(self.plotting)
            start_weights = torchmetrics.utilities.dim_zero_cat(self.start_weights)
            end_weights = torchmetrics.utilities.dim_zero_cat(self.end_weights)
            target = torchmetrics.utilities.dim_zero_cat(self.target)

            # To numpy
            plotting = plotting.cpu().detach().numpy()
            start_weights = start_weights.cpu().detach().numpy().flatten()
            end_weights = end_weights.cpu().detach().numpy().flatten()
            target = target.cpu().detach().numpy().flatten()

        # Else the state tensors already contain numpy, and just need to be concatenated
        else:
            plotting = np.concatenate(self.plotting, axis=0)
            start_weights = np.concatenate(self.start_weights, axis=0).flatten()
            end_weights = np.concatenate(self.end_weights, axis=0).flatten()
            target = np.concatenate(self.target, axis=0).flatten()

        # Isolate source and target weights
        source_weight = end_weights[target == 0]  # Want ending weights for the source
        target_weight = start_weights[target == 1] # And the starting weights for the target

        # If we want to draw plots, do so here
        plot_dict = {}
        if self.draw_plots:
            plot_dict = pu.make_logged_plots(
                plotting, target, start_weights, end_weights, save_location=self.save_location, **kwargs
            )

        # Results list
        results = []

        # Loop through plotting dimensions
        for i, (key, hist_dict) in enumerate(self.hist_info.items()):

            # If we are not using this dimension for W1 computation, continue
            if not hist_dict['w1_eval']:
                continue

            # Slice this dimension
            this_dim = plotting[:,i]

            # Separate into source / target
            source_dist = this_dim[target == 0]
            target_dist = this_dim[target == 1]

            # Calculate 1D wasserstein distance
            this_wass = scipy.stats.wasserstein_distance(source_dist, target_dist, u_weights=source_weight, v_weights=target_weight)

            # Append to results list
            results.append(this_wass)

        # Return the sum over all plotting dimensions, and the plot dictionary
        return np.sum(np.array(results)), plot_dict