#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: <Kyle Dunlap>

Simple Velocity Limit

Evaluates if desired action violates max velocity constraint.
	If so, calculates maximum allowable force that will no violate constraint.
	If not, passes through desired action.

"""

import numpy as np
import os, sys, inspect
import math

class RTA():
	def __init__(self, env):
		# Define parameters
		self.mass_chaser = env.mass_deputy
		self.mean_motion = env.n
		self.max_available_thrust = env.force_magnitude
		self.controller_sample_period = env.tau
		self.controller_sample_rate = 1 / self.controller_sample_period
		self.filter_sample_period = env.tau
		self.filter_sample_rate = 1 / self.filter_sample_period
		self.vel_threshold = env.vel_threshold

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

		# Extract current state data
		x = x0[0]
		y = x0[1]
		x_dot_old = x0[3]
		y_dot_old = x0[4]
		x_force = u_des[0,0]
		y_force = u_des[1,0]

		# Define acceleration functions
		x_acc = (3 * self.mean_motion ** 2 * x) + (2 * self.mean_motion * y_dot_old) + (x_force / self.mass_chaser)
		y_acc = (-2 * self.mean_motion * x_dot_old) + (y_force / self.mass_chaser)
		# Integrate acceleration to calculate velocity
		x_dot = x_dot_old + x_acc * self.controller_sample_period
		y_dot = y_dot_old + y_acc * self.controller_sample_period

		# Check if over max velocity, and return True if it is violating constraint
		rH = np.sqrt(x**2 + y**2) # m, distance between deputy and chief
		vH = np.sqrt(x_dot**2 + y_dot**2) # Velocity Magnitude
		vH_max = 2 * self.mean_motion * rH + self.vel_threshold # Max Velocity

		# If violating, set to True
		if vH > vH_max:
			over_max_vel = True
		else:
			over_max_vel = False

		# Calculate velocity angle
		vtheta = math.atan(y_dot/x_dot)
		if x_dot <0:
			vtheta += math.pi

		# If violating:
		if over_max_vel:
			# Use angle and max velocity to scale down velocity
			x_dot_new = vH_max * math.cos(vtheta)
			y_dot_new = vH_max * math.sin(vtheta)

			# Take derivative
			x_acc = (x_dot_new - x_dot_old) / self.controller_sample_period
			y_acc = (y_dot_new - y_dot_old) / self.controller_sample_period

			# Calculate force required to get acceleration
			x_force = (x_acc - (3 * self.mean_motion ** 2 * x) - (2 * self.mean_motion * y_dot_new)) * self.mass_chaser
			y_force = (y_acc - (-2 * self.mean_motion * x_dot)) * self.mass_chaser


			# Define updated action, set flag to true
			u = np.array([[x_force], [y_force], [0]])

		# If not violating:
		else:
			u = u_des

		return u
