'''
OpenAI SpinningUp's implementation of simplest policy gradient, designed to work
with the dubins aircraft and spacecraft docking environments

Modified by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import aero_gym
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import os
import random

# Used to track actual duration of experiment
start_time = time.time()
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Assumes spacecraftdockingrl is in your home directory
PATH = os.path.expanduser("~") + "/aerospacerl/RL"
if not os.path.isdir(PATH):
	print('PATH ISSUE - UPDATE YOUR PATH')
	exit()

# Builds neura network
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
	# Build a feedforward neural network.
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)

def train(env_name='dubins-aircraft-v0', hidden_sizes=[400,300], lr=1e-4,
		  epochs=125, batch_size=5000, render=False, TensorBoard=True,
		  RandomVelo=False, RandomPos=False, plot=True, plot_every=20,
		  plot2='Force', RTA=False, save_nn=False, save_every=1000, load=False):

	# make environment, check spaces, get obs / act dims
	env = gym.make(env_name)
	assert isinstance(env.observation_space, Box), \
		"This example only works for envs with continuous state spaces."
	assert isinstance(env.action_space, Discrete), \
		"This example only works for envs with discrete action spaces."

	# Use same seed for each experiment
	env.seed(0)

	# Save start time and count episodes
	time0 = time.time()
	total_episodes = 0

	obs_dim = env.observation_space.shape[0]
	n_acts = env.action_space.n

	# make core of policy network
	logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

	if load:
		model = f"{PATH}/saved_models/example.dat"
		logits_net.load_state_dict(torch.load(model))

	# make function to compute action distribution
	def get_policy(obs):
		logits = logits_net(obs)
		return Categorical(logits=logits)

	# make action selection function (outputs int actions, sampled from policy)
	def get_action(obs):
		return get_policy(obs).sample().item()

	# Obtains action with highest probability
	def get_best_action(obs):
		logits = logits_net(obs)
		return np.argmax(logits.detach().numpy())

	# make loss function whose gradient, for the right data, is policy gradient
	def compute_loss(obs, act, weights):
		logp = get_policy(obs).log_prob(act)
		return -(logp * weights).mean()

	# make optimizer
	optimizer = Adam(logits_net.parameters(), lr=lr)

	# Initialize if set to True
	if RandomVelo:
		env.random_velocity = True
	if RandomPos:
		env.random_mountain = True
	if RTA:
		env.init_velocity = 1
		# env.RTA = True
	if plot:
		fig, (ax1,ax2) = plt.subplots(1,2)

	# for training policy
	def train_one_epoch(plot=False, label=None):
		# make some empty lists for logging.
		batch_obs = []          # for observations
		batch_acts = []         # for actions
		batch_weights = []      # for R(tau) weighting in policy gradient
		batch_rets = []         # for measuring episode returns
		batch_lens = []         # for measuring episode lengths
		env.success = 0			# for measuring success rate
		episodes = 0			# for counting episodes
		total_force = []		# for measuring average control input

		# reset episode-specific variables
		obs = env.reset()       # first obs comes from starting distribution
		done = False            # signal from environment that episode is over
		ep_rews = []            # list for rewards accrued throughout ep

		# render first episode of each epoch
		finished_rendering_this_epoch = False

		# collect experience by acting in the environment with current policy
		while True:
			# rendering
			if (not finished_rendering_this_epoch) and render:
				env.render()

			# save obs
			batch_obs.append(obs.copy())

			# act in the environment
			act = get_action(torch.as_tensor(obs, dtype=torch.float32))

			# If RTA is on, check if action violates velocity constraint
			if RTA:
				act_list = env.RTA(act)
				if isinstance(act_list, int) or np.issubdtype(type(act_list), np.integer): # If act_list only has one action
					act = act_list
				else:
					act_prob = []
					# Find associated probability of every acceptable action
					for i in range(len(act_list)):
						act_prob.append(logits_net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()[act_list[i]])
					# Sample from new list of actions
					act = (act_list[Categorical(logits=torch.as_tensor(act_prob)).sample().item()])

			obs, rew, done, _ = env.step(act)

			# save action, reward
			batch_acts.append(act)
			ep_rews.append(rew)

			if done:
				# if episode is over, record info about episode
				ep_ret, ep_len = sum(ep_rews), len(ep_rews)
				batch_rets.append(ep_ret)
				batch_lens.append(ep_len)

				# Episode and control input counters
				episodes += 1
				total_force.append(env.control_input)

				# the weight for each logprob(a|s) is R(tau)
				batch_weights += [ep_ret] * ep_len

				# reset episode-specific variables
				obs, done, ep_rews = env.reset(), False, []

				# won't render again this epoch
				finished_rendering_this_epoch = True

				# end experience loop if we have enough of it
				if len(batch_obs) > batch_size:
					# Calculate success rate
					success_rate = env.success / episodes
					# Run new episode and plot one trajectory using best action
					if plot:
						# Reset variables
						x_ep = []
						y_ep = []
						if epochs == label+1 and env_name=='spacecraft-docking-v0' and plot2=='Force':
							controlx = [0]
							controly = [0]
						elif epochs == label+1 and env_name=='spacecraft-docking-v0' and plot2=='Velocity':
							vH = []
							vH_max = []
						obs = env.reset()
						done = False
						# Run episode using best actions
						while not done:
							act = get_best_action(torch.as_tensor(obs, dtype=torch.float32))
							# If RTA is on, check if action violates velocity constraint
							if RTA:
								act_list = env.RTA(act)
								if isinstance(act_list, int) or np.issubdtype(type(act_list), np.integer): # If act_list only has one action
									act = act_list
								else:
									act_prob = []
									# Find associated probability of every acceptable action
									for i in range(len(act_list)):
										act_prob.append(logits_net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()[act_list[i]])
									# Sample from new list of actions
									act = (act_list[Categorical(logits=torch.as_tensor(act_prob)).sample().item()])
							obs, _, done, _ = env.step(act)
							x_ep.append(obs[0])
							y_ep.append(obs[1])
							if epochs == label+1 and env_name=='spacecraft-docking-v0' and plot2=='Force':
								controlx.append(controlx[len(controlx)-1]+env.x_force)
								controly.append(controly[len(controly)-1]+env.y_force)
							elif epochs == label+1 and env_name=='spacecraft-docking-v0' and plot2=='Velocity':
								vH.append(env.vH)
								vH_max.append(env.vH_max)

						# Plot solid line for last epoch
						if epochs == label+1:
							ax1.plot(x_ep,y_ep, label=label, linestyle='-')
						# Dashed lines for all other epochs
						else:
							ax1.plot(x_ep,y_ep, label=label, linestyle='--')
						# Plot mountain position if aircraft hits random mountain
						if env_name=='dubins-aircraft-v0':
							if RandomPos and (np.sqrt(abs(x_ep[-1]-env.x_mountain) ** 2 + abs(y_ep[-1]-env.y_mountain) ** 2) - env.r_mountain - env.r_aircraft) <= 0:
								ax1.plot(env.x_mountain,env.y_mountain,'kx',markersize=8)
						# Spacecraft subplot 2
						if env_name=='spacecraft-docking-v0' and epochs == label+1 and plot2=='Force':
							ax2.plot(range(len(controlx)),controlx, label='Fx')
							ax2.plot(range(len(controly)),controly, label='Fy')
						elif env_name=='spacecraft-docking-v0' and epochs == label+1 and plot2=='Velocity':
							ax2.plot(range(len(vH)),vH, label='Velocity')
							ax2.plot(range(len(vH_max)),vH_max, 'r', label='Max Velocity')

					break

		# take a single policy gradient update step
		optimizer.zero_grad()
		batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
								  act=torch.as_tensor(batch_acts, dtype=torch.int32),
								  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
								  )
		batch_loss.backward()
		optimizer.step()

		return batch_loss, batch_rets, batch_lens, success_rate, total_force, episodes

	# Create TensorBoard file if True
	if TensorBoard:
		if env_name=='dubins-aircraft-v0':
			Name = f"{PATH}/runs/ac/Dubins-aircraft-" + current_time
		elif env_name=='spacecraft-docking-v0':
			Name = f"{PATH}/runs/sc/Spacecraft-docking-" + current_time
		writer = SummaryWriter(Name)

	try: # Used to break while loop
		while True:
			# training loop
			for i in range(epochs):
				# Plot if last epoch or 'plot_every' epoch
				if i == epochs - 1 or i % plot_every == 0:
					batch_loss, batch_rets, batch_lens, success_rate, total_force, episodes = train_one_epoch(plot=plot, label=i)
				else: # Do not plot
					batch_loss, batch_rets, batch_lens, success_rate, total_force, episodes = train_one_epoch()
				print('epoch: %3d \t return: %7.1f \t ep_len: %6.1f \t success: %2d %% \t force: %6.1f'%
						(i, np.mean(batch_rets), np.mean(batch_lens), success_rate*100, np.mean(total_force)))

				# Save to TensorBoard if True
				if TensorBoard:
					# writer.add_scalar('Loss', batch_loss, i)
					writer.add_scalar('Return', np.mean(batch_rets), i)
					writer.add_scalar('Episode-Length', np.mean(batch_lens), i)
					writer.add_scalar('Success-Rate', success_rate, i)
					writer.add_scalar('Control-Input', np.mean(total_force), i)

				# Episode counter
				total_episodes += episodes

				# save NN every save_every epochs
				if save_nn and i % save_every == 0 and i != 0:
					if not os.path.isdir(f"{PATH}/models"):
						os.mkdir(f"{PATH}/models")
					if env_name=='dubins-aircraft-v0':
						if not os.path.isdir(f"{PATH}/models/ac"):
							os.mkdir(f"{PATH}/models/ac")
						Name2 = f"{PATH}/models/ac/Dubins-aircraft-" + current_time + f"-epoch{i}.dat"
					elif env_name=='spacecraft-docking-v0':
						if not os.path.isdir(f"{PATH}/models/sc"):
							os.mkdir(f"{PATH}/models/sc")
						Name2 = f"{PATH}/models/sc/Spacecraft-docking-" + current_time + f"-epoch{i}.dat"
					torch.save(logits_net.state_dict(), Name2)

			break

	except KeyboardInterrupt:
		# **Press Ctrl-C to terminate training but still plot and save results**
		# Run new episode and plot one trajectory using best action
		if plot:
			# Reset variables
			x_ep = []
			y_ep = []
			if env_name=='spacecraft-docking-v0' and plot2=='Force':
				controlx = [0]
				controly = [0]
			elif env_name=='spacecraft-docking-v0' and plot2=='Velocity':
				vH = []
				vH_max = []
			obs = env.reset()
			done = False
			# Run episode using best actions
			while not done:
				act = get_best_action(torch.as_tensor(obs, dtype=torch.float32))
				# If RTA is on, check if action violates velocity constraint
				if RTA:
					act_list = env.RTA(act)
					if isinstance(act_list, int) or np.issubdtype(type(act_list), np.integer): # If act_list only has one action
						act = act_list
					else:
						act_prob = []
						# Find associated probability of every acceptable action
						for i in range(len(act_list)):
							act_prob.append(logits_net(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()[act_list[i]])
						# Sample from new list of actions
						act = (act_list[Categorical(logits=torch.as_tensor(act_prob)).sample().item()])
				obs, _, done, _ = env.step(act)
				x_ep.append(obs[0])
				y_ep.append(obs[1])
				if env_name=='spacecraft-docking-v0' and plot2=='Force':
					controlx.append(controlx[len(controlx)-1]+env.x_force)
					controly.append(controly[len(controly)-1]+env.y_force)
				elif env_name=='spacecraft-docking-v0' and plot2=='Velocity':
					vH.append(env.vH)
					vH_max.append(env.vH_max)

			ax1.plot(x_ep,y_ep, label=i, linestyle='-')
			# Plot mountain position if aircraft hits random mountain
			if env_name=='dubins-aircraft-v0':
				if RandomPos and (np.sqrt(abs(x_ep[-1]-env.x_mountain) ** 2 + abs(y_ep[-1]-env.y_mountain) ** 2) - env.r_mountain - env.r_aircraft) <= 0:
					ax1.plot(env.x_mountain,env.y_mountain,'kx',markersize=8)
			# Spacecraft subplot 2
			if env_name=='spacecraft-docking-v0' and plot2=='Force':
				ax2.plot(range(len(controlx)),controlx, label='Fx')
				ax2.plot(range(len(controly)),controly, label='Fy')
			elif env_name=='spacecraft-docking-v0' and plot2=='Velocity':
				ax2.plot(range(len(vH)),vH, label='Velocity')
				ax2.plot(range(len(vH_max)),vH_max, 'r', label='Max Velocity')


	# Set up plot if true
	if plot:
		if env_name=='dubins-aircraft-v0':
			# Plot mountain (if not random every time)
			if not RandomPos:
				ax1.plot(env.x_mountain,env.y_mountain,'kx',markersize=8)
			# Plot boundaries
			ax1.vlines(env.x_goal,-env.y_threshold,env.y_threshold,colors='y')
			ax1.vlines(-env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
			ax1.hlines(-env.y_threshold,-env.x_threshold,env.x_goal,colors='r')
			ax1.hlines(env.y_threshold,-env.x_threshold,env.x_goal,colors='r')
		elif env_name=='spacecraft-docking-v0':
			# Plot chief
			ax1.plot(0,0,'kx',markersize=8)
			# Plot boundaries
			ax1.vlines(env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
			ax1.vlines(-env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
			ax1.hlines(-env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
			ax1.hlines(env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
		# Labels/legend
		ax1.set_xlabel('X [ft]')
		ax1.set_ylabel('Y [ft]')
		ax1.grid(True)
		ax1.legend()
		ax1.set_title(f'Trajectory Every {plot_every} Epochs')
		if env_name=='spacecraft-docking-v0' and plot2=='Force':
			ax2.set_xlabel('Time Steps')
			ax2.set_ylabel('Force (N)')
			ax2.grid(True)
			ax2.legend()
			ax2.set_title('Sum of Force Over Time for Last Episode')
		elif env_name=='spacecraft-docking-v0' and plot2=='Velocity':
			ax2.set_xlabel('Time Steps')
			ax2.set_ylabel('Velocity (m/s)')
			ax2.grid(True)
			ax2.legend()
			ax2.set_title('Velocity Over Time for Last Episode')

	# Save neural network
	if save_nn:
		if not os.path.isdir(f"{PATH}/models"):
			os.mkdir(f"{PATH}/models")
		if env_name=='dubins-aircraft-v0':
			if not os.path.isdir(f"{PATH}/models/ac"):
				os.mkdir(f"{PATH}/models/ac")
			Name2 = f"{PATH}/models/ac/Dubins-aircraft-" + current_time + "-final.dat"
		elif env_name=='spacecraft-docking-v0':
			if not os.path.isdir(f"{PATH}/models/sc"):
				os.mkdir(f"{PATH}/models/sc")
			Name2 = f"{PATH}/models/sc/Spacecraft-docking-" + current_time + "-final.dat"
		torch.save(logits_net.state_dict(), Name2)

	# Print statistics on episodes
	print(f"Episodes per hour: {total_episodes/(time.time()-time0)*3600:.0f}, Episodes per epoch: {total_episodes/epochs:.0f}, Epochs per hour: {epochs/(time.time()-time0)*3600:.0f}")

'''
Main training function:
env_name: What environment you are Using
hidden_sizes: hidden layers in neural network
lr: learning rate (used in optimizer)
epochs: groups of episodes trained together (one training per epoch)
batch_size: minimum number of simulated steps per epoch
render: True renders once per epoch, False does not
TensorBoard: True saves data used in TensorBoard (Update your path), False does not
RandomVelo/RandomPos: Use for dubins-aircraft-v0, True allows random aircraft velocity/mountain position
plot/plot_every: plot=True plots trajectory every [plot_every] epochs
plot2: Used to plot either 'Force' or 'Velocity' for spacecraft-docking subplot 2
RTA: Used to turn RTA on or off to constrain velocity to max velocity for spacecraft-docking
save_nn: Saves neural network to specified location
save_every: Saves neural network every _ epochs
load: True to load previous model (update path), False to create new model
'''

train(env_name="spacecraft-docking-v0", render=False, lr=1e-3, epochs=1000, batch_size=5000,
 hidden_sizes=[64, 64], TensorBoard=True, RandomVelo=False, RandomPos=False, plot=True, plot_every=500,
 plot2='Velocity', RTA=False, save_nn=True, save_every=1000, load=False)


# Show experiment duration
print(f"Run Time: {time.time()-start_time:0.4} seconds")

# Show Plot (comment out if plot=False)
plt.show()

'''
** To start TensorBoard, run the following command in your terminal with your specific path to aerospacerl:**
Aircraft:
tensorboard --logdir aerospacerl/RL/runs/ac
Spacecraft:
tensorboard --logdir aerospacerl/RL/runs/sc
'''
