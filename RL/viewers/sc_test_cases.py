'''
Runs test cases
Used with Spacecraft Docking Environment

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

plt.rcParams.update({"text.usetex": True,'font.size': 20, 'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']})
plt.rcParams.update({'figure.autolayout': True})

def main(Test='NoRTA', TrainingCases=['NoRTA'], RANGE=1000, ac_kwargs=dict(hidden_sizes=[64,64])):

	"""
	Test: Test Case
		'NoRTA', 'ExS', 'ImS', 'ExO', or 'ImO'
	Train: Training Case
		'NoRTA', 'NoRTAHP', 'ExS', 'ImS', 'ExO', or 'ImO'
	RANGE: Test Range (m)
		1000 or 10000
	ac_kwargs: Neural network parameters
		dictionary with hidden layers, ex: dict(hidden_sizes=[64,64]
	NN MODELS (below): Saved trained models
	"""

	##### NN MODELS #####
	NoRTA_model = "NoRTA.dat"
	NoRTAHP_model = "NoRTAHP.dat"
	ExS_model = "ExS.dat"
	ImS_model = "ImS.dat"
	ExO_model = "ExO.dat"
	ImO_model = "ImO.dat"
	#####################

	env = gym.make('spacecraft-docking-continuous-v0')

	# Defines test points
	if RANGE == 10000:
		Distance = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
	elif RANGE == 1000:
		Distance = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

	Angle = [1.57, 5.5, 4.71, 2.36, 3.93, 0.79, 1.18, 3.14, 4.32, 0]
	Vx = [-0.1, -0.25, 0.25, 0.1, -0.5, 0.5, -0.75, 0.75, -1, 1]
	Vy = [0.1, -0.1, -0.25, 0.25, 0.5, -0.5, -0.75, 1, 0.75, -1]

	if Test == 'ExS':
		from Explicit_Switching import RTA
	elif Test == 'ImS':
		from Implicit_Switching import RTA
	elif Test == 'ExO':
		from Explicit_Optimization import RTA
	elif Test == 'ImO':
		from Implicit_Optimization import RTA

	if Test == 'ExS' or Test == 'ImS' or Test == 'ExO' or Test == 'ImO':
		# Call RTA class
		rta = RTA(env)

		# Define action
		def RTA_act(obs, act):
			# Clip action to be within accepted range
			act = np.clip(act, -env.force_magnitude, env.force_magnitude)
			# Rearrange observation state vector
			x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
			# Rearrange action vector
			u_des = np.array([[act[0]], [act[1]], [0]])
			# Call RTA function
			u = rta.main(x0, u_des)
			# Extract relevant data
			new_act = [u[0,0], u[1,0]]
			# Determine if RTA adjusted action
			if np.sqrt((act[0] - new_act[0])**2 + (act[1] - new_act[1])**2) < 0.0001:
				# Set flag for tracking/reward function
				env.RTA_on = False
			else:
				env.RTA_on = True
			# Return new action
			return new_act

	for Train in TrainingCases:
		# Load neural network
		ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
		# Load appropriate model
		if Train == 'NoRTA':
			ac.load_state_dict(torch.load(f"{PATH}/{NoRTA_model}"))
		elif Train == 'NoRTAHP':
			ac.load_state_dict(torch.load(f"{PATH}/{NoRTAHP_model}"))
		elif Train == 'ExS':
			ac.load_state_dict(torch.load(f"{PATH}/{ExS_model}"))
		elif Train == 'ImS':
			ac.load_state_dict(torch.load(f"{PATH}/{ImS_model}"))
		elif Train == 'ExO':
			ac.load_state_dict(torch.load(f"{PATH}/{ExO_model}"))
		elif Train == 'ImO':
			ac.load_state_dict(torch.load(f"{PATH}/{ImO_model}"))

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
			if RANGE == 1000 and Test != 'ImS':
				env.max_control = 750
			if RANGE == 1000 and Test == 'ImS':
				env.max_control = 4000
			if RANGE == 10000:
				env.max_time = 10000

			# Run episode
			while not done:
				# Get best action
				act = get_best_action(obs)
				# Pass through RTA
				if Test == 'ExS' or Test == 'ImS' or Test == 'ExO' or Test == 'ImO':
					act = RTA_act(obs,act)
				# Take step in environment
				obs, _, done, _ = env.step(act)
				# Track if velocity violated constraint (No RTA)
				if Test == 'NoRTA':
					over_max_vel, _, _ = env.check_velocity(act[0], act[1])
					if over_max_vel:
						RTA_percent += 1
				# Track if RTA is on
				elif Test == 'ExS' or Test == 'ImS' or Test == 'ExO' or Test == 'ImO':
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
			if Train == 'NoRTAHP':
				dash = 'r'
			elif Train == 'NoRTA':
				dash = 'tab:orange'
			elif Train == 'ExS':
				dash = 'y'
			elif Train == 'ImS':
				dash = 'tab:green'
			elif Train == 'ExO':
				dash = 'b'
			elif Train == 'ImO':
				dash = 'tab:purple'
			plt.plot(rH,vH,dash)
			plt.figure(2)
			plt.plot(x,y,dash)

		# Print RTA on percentage
		print(f"{Train} Average RTA % On: {RTA_percent/steps*100:.1f} %")

	# Plot setup
	plt.figure(1)
	plt.plot([0, 10000],[0.2, 20.74], '--', color='black',label='Max Velocity Limit')
	plt.plot([0, 10000],[-0.2, 4.935], '--', color='coral',label='Min Velocity Limit')
	# plt.title('Velocity vs. Position')
	if RANGE == 1000:
		plt.ylim([0, 2.5])
		plt.xlim([0, 1200])
	elif RANGE == 10000:
		plt.xlim([0, 10000])
		plt.ylim([0, 20])
	plt.xlabel('Relative Position ($\Vert \pmb{r}_{\mathrm{H}} \Vert$) [m]')
	plt.ylabel('Relative Velocity ($\Vert \pmb{v}_{\mathrm{H}} \Vert$) [m/s]')
	# plt.legend()
	plt.grid(True)
	plt.fill_between([0,15000],0,[0.2,0.2+2*0.001027*15000],color=(244/255,249/255,241/255))
	plt.fill_between([0,15000],[0.2,0.2+2*0.001027*15000],[1000,1000],color=(255/255,239/255,239/255))

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
	plt.xlabel('$x$ [km]')
	plt.ylabel('$y$ [km]')

	plt.figure(3)
	plt.plot(0,0,color='r', linewidth=2)
	plt.plot(0,0,color='tab:orange', linewidth=2)
	plt.plot(0,0,color='y', linewidth=2)
	plt.plot(0,0,color='tab:green', linewidth=2)
	plt.plot(0,0,color='b', linewidth=2)
	plt.plot(0,0,color='tab:purple', linewidth=2)
	plt.plot(0,0,'--',color='black')
	plt.plot(0,0,'--',color='coral')
	plt.axis('off')
	plt.legend(['No RTA - HP','No RTA','Explicit Switching','Implicit Switching','Explicit Optimization','Implicit Optimization','Max Velocity Limit','Min Velocity Limit'], loc='upper center')
	# plt.legend(['No RTA - HP','Explicit Switching','Implicit Switching','Explicit Optimization','Implicit Optimization','Max Velocity Limit','Min Velocity Limit'], loc='upper center')

	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', type=str, default='NoRTA') # Training Case: 'NoRTA', 'ExS', 'ImS', 'ExO', or 'ImO'
	parser.add_argument('--train', nargs='+', default=['NoRTAHP','NoRTA','ExS','ImS','ExO','ImO']) # Training Case: 'NoRTA', 'NoRTAHP', 'ExS', 'ImS', 'ExO', or 'ImO'
	parser.add_argument('--range', type=int, default=1000) # Test Range: 1000 or 10000
	parser.add_argument('--hid', type=int, default=64) # Hidden layer nodes
	parser.add_argument('--l', type=int, default=2) # Number of hidden layers
	args = parser.parse_args()

	main(Test=args.test, TrainingCases=args.train, RANGE=args.range, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))

'''
Example of how to run sc_test_cases in terminal from home directory, testing with Explicit Switching and using model trained with Implicit Optimization:
python aerospacerl/RL/viewers/sc_test_cases.py --test ExS --train ImO
'''
