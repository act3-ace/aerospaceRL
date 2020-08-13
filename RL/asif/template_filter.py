#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: <your name here>

A template for creating RTA filters, passes desired input through unaltered

The main loop will call the ASIF's "main" class and expect
a control signal of appropriate size and type to be returned

In order for this script to work correctly with the main script, avoid
changing the name of the "ASIF" class, or the inputs and outputs of
the "main" function.

"""

import numpy as np


class ASIF(): 
    def __init__(self, env):
        self.mass_chaser = env.mass_deputy
		self.mean_motion = env.n
		self.max_available_thrust = env.force_magnitude
		self.controller_sample_period = env.tau
		self.controller_sample_rate = 1 / self.controller_sample_period
		self.filter_sample_period = env.tau
		self.filter_sample_rate = 1 / self.filter_sample_period

        self.zero_input = np.zeros([3,1])

        ######################################################################
        # Set up the ASIF parameters and options here
        ######################################################################


    def main(self, x0, u_des):
        """
        Parameters
        ----------
        x : numpy array with 6 elements
            x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
            where x,y,z are hill frame coordinates
                u : 3x1 numpy array
        u_des = [[Fx], [Fy], [Fz]]
            desired control input

        Returns
        -------
        u : 3x1 numpy array
            u = [[Fx], [Fy], [Fz]]
            returned control input
        """

        ######################################################################
        # Insert your code here
        ######################################################################

        u = u_des

        return u

    def h_s(self, x):
        """
        h_s(x) >= 0 defines the set of all "safe states". The goal of the ASIF
        is to ensure that this constraint remains satisfied

        """
