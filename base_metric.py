""" base_metric.py - This file contains the BaseMetric class, which
is a base class for OF classifier performance metric.

This base class is necessary because we implement all of the plotting callbacks
for a regular training as a subclass of torchmetrics.Metric. The reason is that this
class has built-in functionality for aggregating model inputs / outputs across
many batches and devices. This is necessary for calculating performance metrics,
but also for building whatever plots we might be interested in at a given point
in training.

Author: Kevin Greif
Last updated 05/08/2024
python3
"""

import torchmetrics
import numpy as np

import utils.plotting_utils as pu


class BaseMetric(torchmetrics.Metric):
    """A torchmetrics metric subclass which acts as the base class that handles
    all of the plotting.
    """

    def __init__(self, hist_info, draw_plots=False, save_location=None, **kwargs):
        """__init__ - Init function for the class. Both definees the state of the
        metric, and accepts some keyword arguments to control plot drawing.

        Arguments:
        hist_info - Dictionary describing the histograms used in the plotting
        draw_plots - Optional, default False. If True, will draw plots of the plo
            dimensions.
        If false then this class essentially acts as a wrapper for calculating whateve
            performance metric defined in the child class.
        save_location - Optional, default None. If provided, will save the plots t
            this location

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
        """update - Function to update the metric state on each batch."""
        self.plotting.append(plotting)
        self.start_weights.append(start_weights)
        self.end_weights.append(end_weights)
        self.target.append(labels)

    def compute(self, from_torch=True, **kwargs):
        """compute - Actually produce the plots. This function should be overloaded
        with an appropriate call to "super" in for any subclass.

        Arguments:
        from_torch - Optional, default True. If True, will convert the state tensors
            to numpy. If false assumes state tensors are already numpy
        Returns: A dictionary of plots with form
            {plot_name: stored location of .png file}.
        If draw_plots is False, then this dictionary is empty.
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

        # Make instance variables
        self.plotting = plotting
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.target = target

        # If we want to draw plots, do so here
        plot_dict = {}
        if self.draw_plots:
            plot_dict = pu.make_logged_plots(
                plotting,
                target,
                start_weights,
                end_weights,
                save_location=self.save_location,
                **kwargs
            )

        # Return the plot dictionary
        return plot_dict

    def reset(self):
        """reset - Reset the metric state."""
        self.plotting = []
        self.start_weights = []
        self.end_weights = []
        self.target = []

        super().reset()
