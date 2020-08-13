'''
Runs test cases

Created by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

import gym
import aero_gym
import os
import math
import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'spinup_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RunTimeAssurance'))
PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
import core

plt.rcParams.update({'font.size': 16}) # For Presentation/Paper
plt.rcParams.update({'figure.autolayout': True})

def main(Test='NoRTA', Train='NoRTA', RANGE=1000, ac_kwargs=dict(hidden_sizes=[64,64])):

	"""
	Test: Test Case
		'NoRTA', 'Velocity', 'Simplex', or 'IASIF'
	Train: Training Case
		'NoRTA', 'Velocity', 'Simplex', or 'IASIF'
	RANGE: Test Range (m)
		1000 or 10000
	ac_kwargs: Neural network parameters
		dictionary with hidden layers, ex: dict(hidden_sizes=[64,64]
	NN MODELS (below): Saved trained models
	"""

	##### NN MODELS #####
	NoRTA_model = "NoRTA1.dat"
	Velocity_model = "Velocity1.dat"
	Simplex_model = "ISimplex1.dat"
	IASIF_model = "IASIF1.dat"
	#####################

	env = gym.make('spacecraft-docking-continuous-v0')
	env.tau = 1 # Change to 1 sec time step

	# Defines test points
	if RANGE == 10000:
		Distance = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
	elif RANGE == 1000:
		Distance = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

	Angle = [1.57, 5.5, 4.71, 2.36, 3.93, 0.79, 1.18, 3.14, 4.32, 0]
	Vx = [-0.1, -0.25, 0.25, 0.1, -0.5, 0.5, -0.75, 0.75, -1, 1]
	Vy = [0.1, -0.1, -0.25, 0.25, 0.5, -0.5, -0.75, 1, 0.75, -1]

	# Import ASIF
	if Test == 'Velocity':
		from Simple_velocity_limit import RTA
	elif Test == 'Simplex':
		from ISimplex import RTA
	elif Test == 'IASIF':
		from IASIF import RTA

	if Test == 'Velocity' or Test == 'Simplex' or Test == 'IASIF':
		# Call ASIF class
		rta = RTA(env)

		# Define action
		def RTA_act(obs, act):
			# Clip action to be within accepted range
			act = np.clip(act, -env.force_magnitude, env.force_magnitude)
			# Rearrange observation state vector
			x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
			# Rearrange action vector
			u_des = np.array([[act[0]], [act[1]], [0]])
			# Call asif function
			u = rta.main(x0, u_des)
			# Extract relevant data
			new_act = [u[0,0], u[1,0]]
			# Determine if RTA adjusted action
			if abs(np.sqrt(new_act[0]**2+new_act[1]**2) - np.sqrt(act[0]**2+act[1]**2)) < 0.0001:
				# Set flag for tracking/reward function
				env.RTA_on = False
			else:
				env.RTA_on = True
			# Return new action
			return new_act

	# Load neural network
	ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
	# Load appropriate model
	if Train == 'NoRTA':
		ac.load_state_dict(torch.load(f"{PATH}/{NoRTA_model}"))
	elif Train == 'Velocity':
		ac.load_state_dict(torch.load(f"{PATH}/{Velocity_model}"))
	elif Train == 'Simplex':
		ac.load_state_dict(torch.load(f"{PATH}/{Simplex_model}"))
	elif Train == 'IASIF':
		ac.load_state_dict(torch.load(f"{PATH}/{IASIF_model}"))

	# Use best action (mean of policy's probability distribution)
	def get_best_action(obs):
		with torch.no_grad():
			act = ac.pi.mu_net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
		return act

	# Set variables
	env.termination_condition = True # Prints cause of termination
	RTA_percent = 0 # Tracks percentage of time RTA is on
	steps = 0 # Tracks number of steps

	# for 10 test points
	for i2 in range(len(Distance)):
		# Reset variables
		done = False
		env.reset()
		# Used to track trajectories for plots
		rH = []
		vH = []
		x = []
		y = []

		# Reset environment conditions for each test case
		theta = Angle[i2]
		env.position_deputy = Distance[i2]
		env.x_deputy = env.position_deputy*math.cos(theta)
		env.y_deputy = env.position_deputy*math.sin(theta)
		x_dot = Vx[i2]
		y_dot = Vy[i2]
		env.rH = env.position_deputy
		env.state = np.array([env.x_deputy, env.y_deputy, x_dot, y_dot])
		obs = env.state
		env.x_threshold = 1.5 * env.position_deputy
		env.y_threshold = 1.5 * env.position_deputy
		env.max_time = 10000
		env.max_control = 10000

		# Run episode
		while not done:
			# Get best action
			act = get_best_action(obs)
			# Pass through RTA
			if Test == 'Velocity' or Test == 'IASIF' or Test == 'Simplex':
				act = RTA_act(obs,act)
			# Take step in environment
			obs, _, done, _ = env.step(act)
			# Track if velocity violated constraint (No RTA)
			if Test == 'NoRTA':
				over_max_vel, _, _ = env.check_velocity(act[0], act[1])
				if over_max_vel:
					RTA_percent += 1
			# Track if RTA is on
			elif Test == 'Velocity' or Test == 'IASIF' or Test == 'Simplex':
				if env.RTA_on:
					RTA_percent += 1
			steps += 1

			# Track for plotting
			rH.append(env.rH)
			vH.append(env.vH)
			x.append(obs[0])
			y.append(obs[1])

		# Plot trajectories
		plt.figure(1)
		plt.plot(rH,vH)
		plt.figure(2)
		plt.plot(x,y)

	# Print RTA on percentage
	print(f"Average RTA % On: {RTA_percent/steps*100:.1f} %")

	# Plot setup
	plt.figure(1)
	plt.plot([0, 10000],[0.2, 20.74], '--r', label='Max Velocity Limit')
	plt.plot([0, 10000],[-0.2, 4.935], '--k', label='Min Velocity Limit')
	# plt.title('Velocity vs. Position')
	if RANGE == 1000:
		plt.xlim([0, 1200])
		plt.ylim([0, 2.5])
	elif RANGE == 10000:
		plt.xlim([0, 10000])
		plt.ylim([0, 20])
	plt.xlabel('Distance from Chief (m)')
	plt.ylabel('Relative Velocity (m/s)')
	plt.legend()
	plt.grid(True)

	plt.figure(2)
	# plt.title('Trajectories')
	if RANGE == 1000:
		plt.xlim([-1200, 1200])
		plt.ylim([-1200, 1200])
	elif RANGE == 10000:
		plt.xlim([-11000, 11000])
		plt.ylim([-11000, 11000])
	plt.plot(0,0,'k*', ms=10)
	plt.grid(True)
	plt.xlabel('X position (m)')
	plt.ylabel('Y position (m)')

	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', type=str, default='NoRTA') # Test Case: 'NoRTA', 'Velocity', 'Simplex', or 'IASIF'
	parser.add_argument('--train', type=str, default='NoRTA') # Training Case: 'NoRTA', 'Velocity', 'Simplex', or 'IASIF'
	parser.add_argument('--range', type=int, default=1000) # Test Range: 1000 or 10000
	parser.add_argument('--hid', type=int, default=64) # Hidden layer nodes
	parser.add_argument('--l', type=int, default=2) # Number of hidden layers
	args = parser.parse_args()

	main(Test=args.test, Train=args.train, RANGE=args.range, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))
