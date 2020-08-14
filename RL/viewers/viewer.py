'''
View or Plot using saved neural networks. Include args '--Render' or '--Plot' to render/plot

Created by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

import torch
import torch.nn as nn
import gym
import aero_gym
import numpy as np
import os
import glob
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'spinup_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RunTimeAssurance'))

# Loads specified neural network data
def model(env_name, hidden_sizes, latest, algo, Path):

	"""
	env_name: Environment name
	hidden_sizes: Hidden layers/nodes for neural network
	latest: Load latest model if true
	algo: algorithm used for training model
	Path: path to custom model
	"""

	env = gym.make(env_name)

	if latest == True:
		if algo == 'VPG':
			PATH2 = os.path.join(os.path.dirname(__file__), '..', 'models/VPG')
		elif algo == 'PPO':
			PATH2 = os.path.join(os.path.dirname(__file__), '..', 'models/PPO')

		# Uses latest model
		print(PATH2)
		models = glob.glob(f"{PATH2}/*")
		latest_model = max(models, key=os.path.getctime)

	else:
		# Custom file defined below
		latest_model = Path

	if algo == 'VPG':
		# Define Neural Network
		def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
			# Build a feedforward neural network.
			layers = []
			for j in range(len(sizes)-1):
				act = activation if j < len(sizes)-2 else output_activation
				layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
			return nn.Sequential(*layers)

		obs_dim = env.observation_space.shape[0]
		n_acts = env.action_space.n

		# Create Neural Network
		logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
		logits_net.load_state_dict(torch.load(latest_model))

		net = logits_net

	elif algo == 'PPO':
		import core

		# Define Neural Network
		ac_kwargs = dict(hidden_sizes=hidden_sizes)
		ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
		ac.load_state_dict(torch.load(latest_model))

		net = ac

	return net

# Used to render episodes
def view(env_name='spacecraft-docking-continuous-v0', hidden_sizes=[64,64], episodes=10, latest=False, algo='PPO', RTA_type='off', Path=None, render_every=2, custom_settings=None):

	"""
	env_name: Environment name
	hidden_sizes: Hidden layers/nodes for neural network
	episodes: Number of episodes to render
	latest: Load latest model if true
	algo: algorithm used for training model
	RTA_type: type of RTA to use
	Path: path to custom model
	render_every: render every _ steps
	custom_settings: For GUI
	"""

	# Load neural network
	net = model(env_name, hidden_sizes, latest, algo, Path)

	# Set environment
	env = gym.make(env_name)

	if env_name == 'spacecraft-docking-v0' or env_name == 'spacecraft-docking-continuous-v0':
		if custom_settings is None: #if no settings given, pass in defaults
			#c = (thrust='Block', trace=5, v_arrow=True, f_arrow=True, stars=200, a1=0, b1=0, a2=0, b2=0, e_qualtity=0)
			c = ("Block", 5, True, True, 200, 0, 0, 0, 0, 0, 0)
		else:
			c = custom_settings
		env.thrustVis = c[0] #what type of thrust visualization to use. 'Particle', 'Block', 'None'
		env.trace = c[1] #spacing between trace dots
		env.velocityArrow = c[2] #if velocity arrow is shown
		env.forceArrow = c[3] #if force arrow is shown
		env.stars = c[4] #sets number of stars
		env.ellipse_a1 = c[5] #m
		env.ellipse_b1 = c[6] #m
		env.ellipse_a2 = c[7] #m
		env.ellipse_b2 = c[8] #m
		env.ellipse_quality = c[9] #1/x * pi

	# Define best action for specified algorithm
	if algo == 'VPG':
		def get_best_action(obs):
			logits = net(torch.as_tensor(obs, dtype=torch.float32))
			return np.argmax(logits.detach().numpy())
	elif algo == 'PPO':
		def get_best_action(obs):
			with torch.no_grad():
				act = net.pi.mu_net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
			return act

		# Set RTA
		if RTA_type == 'CBF':
			from CBF_for_speed_limit import RTA
		elif RTA_type == 'Velocity':
			from Simple_velocity_limit import RTA
		elif RTA_type == 'IASIF':
			from IASIF import RTA
		elif RTA_type == 'ISimplex':
			from ISimplex import RTA

		# Initialize RTA variables if on
		if RTA_type != 'off':
			rta = RTA(env)
			env.RTA_reward = RTA_type

			def RTA_act(obs, act):
				act = np.clip(act, -env.force_magnitude, env.force_magnitude)
				x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
				u_des = np.array([[act[0]], [act[1]], [0]])
				u = rta.main(x0, u_des)
				new_act = [u[0,0], u[1,0]]
				if abs(np.sqrt(new_act[0]**2+new_act[1]**2) - np.sqrt(act[0]**2+act[1]**2)) < 0.0001:
					env.RTA_on = False
				else:
					env.RTA_on = True
				return new_act

	# Run episodes
	env.termination_condition = True
	for episode in range(episodes):
		# Reset variables
		done = False
		Rewards = 0
		obs = env.reset()
		RTA_percent = 0
		steps = 0
		# Run episode
		while not done:
			act = get_best_action(obs)
			if RTA_type != 'off' and algo == 'PPO':
				act = RTA_act(obs, act)
			elif RTA_type != 'off' and algo == 'VPG':
				act = env.RTA(act)
				if not isinstance(act, int) and not np.issubdtype(type(act), np.integer):
					act_prob = []
					# Find associated probability of every acceptable action
					for i in range(len(act)):
						act_prob.append(net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()[act[i]])
					# Sample from new list of actions
					act = (act[Categorical(logits=torch.as_tensor(act_prob)).sample().item()])

			# Take step with appropriate action
			obs, rew, done, info = env.step(act)
			Rewards += rew
			steps +=1

			# Render every 'render_every' steps
			if steps % render_every == 0:
				env.render()
			if env.RTA_on:
				RTA_percent += 1

		# Print episode stats
		print(f"View Episode: {episode} \t Rewards: {Rewards:5.2f} \t Time (sec): {env.steps*env.tau:5.1f} \t RTA On %: {RTA_percent/(env.steps+1)*100:2.1f}")

	env.close()

# Used for plotting
def plot(env_name='spacecraft-docking-continuous-v0', hidden_sizes=[64,64], episodes=10, latest=False, algo='PPO', RTA_type='off', Path=None, position=True, velocity=True, force=True, vel_pos=True):

	"""
	Currently only for Spacecraft Docking Environment
	env_name: Environment name
	hidden_sizes: Hidden layers/nodes for neural network
	episodes: Number of episodes to render
	latest: Load latest model if true
	algo: algorithm used for training model
	RTA_type: type of RTA to use
	Path: path to custom model
	position: plots trajectory if true
	velocity: plots velocity vs. time if true
	force: plots control input vs. time if true
	vel_pos: plots velocity vs. position if true
	"""

	# Load neural network data
	net = model(env_name, hidden_sizes, latest, algo, Path)

	# Setup environment
	env = gym.make(env_name)

	# Define best action for specified algorithm
	if algo == 'VPG':
		def get_best_action(obs):
			logits = net(torch.as_tensor(obs, dtype=torch.float32))
			return np.argmax(logits.detach().numpy())
	elif algo == 'PPO':
		def get_best_action(obs):
			with torch.no_grad():
				act = net.pi.mu_net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
			return act

		# Set RTA
		if RTA_type == 'CBF':
			from CBF_for_speed_limit import RTA
		elif RTA_type == 'Velocity':
			from Simple_velocity_limit import RTA
		elif RTA_type == 'IASIF':
			from IASIF import RTA
		elif RTA_type == 'ISimplex':
			from ISimplex import RTA

		# Initialize RTA variables if on
		if RTA_type != 'off':
			rta = RTA(env)
			env.RTA_reward = RTA_type

			def RTA_act(obs, act):
				act = np.clip(act, -env.force_magnitude, env.force_magnitude)
				x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
				u_des = np.array([[act[0]], [act[1]], [0]])
				u = rta.main(x0, u_des)
				new_act = [u[0,0], u[1,0]]
				if abs(np.sqrt(new_act[0]**2+new_act[1]**2) - np.sqrt(act[0]**2+act[1]**2)) < 0.0001:
					env.RTA_on = False
				else:
					env.RTA_on = True
				return new_act

	# Setup plot
	ax = plt.gca()
	# Run episodes
	env.termination_condition = True
	start_time = time.time()
	for episode in range(episodes):
		# Reset variables
		done = False
		obs = env.reset()
		Rewards = 0
		rh = []
		x = []
		y = []
		vH = []
		controlx = [0]
		controly = [0]
		RTA_percent = 0
		vH_max = []
		# Run episode
		while not done:
			act = get_best_action(obs)
			if RTA_type != 'off' and algo == 'PPO':
				act = RTA_act(obs, act)
			elif RTA_type != 'off' and algo == 'VPG':
				act = env.RTA(act)
				if not isinstance(act, int) and not np.issubdtype(type(act), np.integer):
					act_prob = []
					# Find associated probability of every acceptable action
					for i in range(len(act)):
						act_prob.append(net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()[act[i]])
					# Sample from new list of actions
					act = (act[Categorical(logits=torch.as_tensor(act_prob)).sample().item()])

			# Take step with appropriate action
			obs, rew, done, _ = env.step(act)
			Rewards += rew
			# Track trajectories
			x.append(obs[0])
			y.append(obs[1])
			controlx.append(controlx[len(controlx)-1]+env.x_force)
			controly.append(controly[len(controly)-1]+env.y_force)
			vH.append(env.vH)
			rh.append(env.rH)
			vH_max.append(env.vH_max)
			if env.RTA_on:
				RTA_percent += 1
				# plt.figure(2)
				# plt.plot(env.steps,env.vH,'rx')
				# plt.figure(4)
				# plt.plot(env.rH,env.vH,'rx')

		# Print episode stats
		print(f"Plot Episode: {episode} \t Rewards: {Rewards:5.2f} \t Time (sec): {env.steps*env.tau:5.1f} \t RTA On %: {RTA_percent/(env.steps+1)*100:2.1f}")
		print(f"Run time: {time.time()-start_time:.2f}")

		# Plot position, velocity, force trajectories
		color = next(ax._get_lines.prop_cycler)['color']
		if position:
			plt.figure(1)
			plt.plot(x,y, label=f"Episode {episode+1}", color=color)
		if velocity:
			plt.figure(2)
			plt.plot(range(len(vH)),vH, label=f"Velocity, Episode {episode+1}", color=color, linestyle='-')
			plt.plot(range(len(vH_max)), vH_max, label=f"Max Velocity, Episode {episode+1}", color=color, linestyle='--')
		if force:
			plt.figure(3)
			plt.plot(range(len(controlx)),controlx, label=f"Fx, Episode {episode+1}", color=color, linestyle='-')
			plt.plot(range(len(controly)),controly, label=f"Fy, Episode {episode+1}", color=color, linestyle='--')
		if vel_pos:
			plt.figure(4)
			plt.plot(rh,vH, label=f"Episode {episode+1}", color=color, linestyle='-')

	# Finalize plots
	if position:
		plt.figure(1)
		# Plot chief
		plt.plot(0,0,'kx',markersize=8)
		# Plot boundaries
		plt.vlines(env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
		plt.vlines(-env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
		plt.hlines(-env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
		plt.hlines(env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
		plt.xlabel('X [m]')
		plt.ylabel('Y [m]')
		plt.grid(True)
		plt.legend()
		plt.title('Trajectories')
	if velocity:
		plt.figure(2)
		plt.xlabel('Time Steps')
		plt.ylabel('Velocity (m/s)')
		plt.grid(True)
		plt.legend()
		plt.title('Velocity Over Time')
	if force:
		plt.figure(3)
		plt.xlabel('Time Steps')
		plt.ylabel('Force (N)')
		plt.grid(True)
		plt.legend()
		plt.title('Sum of Force Over Time')
	if vel_pos:
		plt.figure(4)
		plt.xlabel('Position (m)')
		plt.ylabel('Velocity (m/s)')
		plt.grid(True)
		plt.plot([0, 10000],[0.2, 20.74], 'r--', label='Max Velocity Limit')
		plt.xlim([0, env.position_deputy*1.2])
		plt.ylim([0, 2*env.n*env.position_deputy*1.2+0.2])
		plt.legend()

	plt.show()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--Render', default=False, action='store_true') # Add arg '--Render' ro render episodes
	parser.add_argument('--Plot', default=False, action='store_true') # Add arg '--Plot' to plot episodes
	parser.add_argument('--env', type=str, default='spacecraft-docking-continuous-v0') # Environment Name
	parser.add_argument('--algo', type=str, default='PPO') # Algorithm used in training
	parser.add_argument('--episodes', type=int, default=1) # Number of episodes to display
	parser.add_argument('--LoadLatest', default=False, action='store_true') # Load NN - Add arg '--LoadLatest' to load last saved model instead of custom model
	parser.add_argument('--hid', type=int, default=64) # Number of hidden layer nodes
	parser.add_argument('--l', type=int, default=2) # Number of hidden layers
	parser.add_argument('--RTA', type=str, default='off') # Run Time Assurance - 4 options: 'CBF', 'Velocity', 'IASIF', or 'ISimplex'
	parser.add_argument('--render_every', type=int, default=4) # Render every _ steps
	parser.add_argument('--PlotTypes', nargs='+', default=['position','vel_pos','velocity','force']) # Which plots to show
	parser.add_argument('--model', type=str, default='Velocity1') # Model from 'saved_models' folder
	args = parser.parse_args()

	PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
	custom_file = f"{PATH}/{args.model}.dat"

	if args.Render:
		view(env_name=args.env, episodes=args.episodes, latest=args.LoadLatest, algo=args.algo, RTA_type=args.RTA, Path=custom_file, render_every=args.render_every, hidden_sizes=[args.hid]*args.l)

	"""
	Plotting Currently Only for Spacecraft Docking Environment
	"""

	if args.Plot:
		position, velocity, force, vel_pos = False, False, False, False
		for arg in args.PlotTypes:
			if arg == 'position':
				position = True
			elif arg == 'velocity':
				velocity = True
			elif arg == 'force':
				force = True
			elif arg == 'vel_pos':
				vel_pos = True

		plot(env_name=args.env, episodes=args.episodes, latest=args.LoadLatest, algo=args.algo, RTA_type=args.RTA, Path=custom_file, hidden_sizes=[args.hid]*args.l, position=position, velocity=velocity, force=force, vel_pos=vel_pos)


'''
Example of how to run viewer in terminal from home directory, plot and render 2 episodes:
python aerospacerl/RL/viewers/viewer.py --Render --Plot --episodes 2
'''
