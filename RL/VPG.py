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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import os
import random

# Used to track actual duration of experiment
start_time = time.time()
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

PATH = os.path.dirname(__file__)

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
		  save_nn=False, save_every=1000, RTA = False):

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

	# make function to compute action distribution
	def get_policy(obs):
		logits = logits_net(obs)
		return Categorical(logits=logits)

	# make action selection function (outputs int actions, sampled from policy)
	def get_action(obs):
		return get_policy(obs).sample().item()

	# make loss function whose gradient, for the right data, is policy gradient
	def compute_loss(obs, act, weights):
		logp = get_policy(obs).log_prob(act)
		return -(logp * weights).mean()

	# make optimizer
	optimizer = Adam(logits_net.parameters(), lr=lr)

	# for training policy
	def train_one_epoch():
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
		# **Press Ctrl-C to terminate training**
		pass

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
save_nn: Saves neural network to specified location
save_every: Saves neural network every _ epochs
RTA: RTA is on if True
'''

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='spacecraft-docking-v0') # Environment Name
	parser.add_argument('--hid', type=int, default=64) # Hidden layer nodes
	parser.add_argument('--l', type=int, default=2) # Number of hidden layers
	parser.add_argument('--lr', type=float, default=1e-3) # Learning Rate
	# parser.add_argument('--seed', '-s', type=int, default=0) # Seed for randomization
	parser.add_argument('--steps', type=int, default=5000) # Steps per epoch (Defaults to enough to run at least one episode per core)
	parser.add_argument('--epochs', type=int, default=1000) # Number of epochs
	parser.add_argument('--NoTB', default=True, action='store_false') # Log to TnesorBoard - Add arg '--NoTB' if you don't want to log to TensorBoard
	parser.add_argument('--NoSave', default=True, action='store_false') # Save NN - Add arg '--NoSave' if you don't want to save NN
	parser.add_argument('--SaveEvery', type=int, default=500) # Save NN every _ epochs
	parser.add_argument('--RTA', default=False, action='store_true') # RTA, on or off for simple velocity limit
	parser.add_argument('--render', default=False, action='store_true') # Render if True
	args = parser.parse_args()

train(env_name=args.env, render=args.render, lr=args.lr, epochs=args.epochs, batch_size=args.steps,
 hidden_sizes=[args.hid]*args.l, TensorBoard=args.NoTB, save_nn=args.NoSave, save_every=args.SaveEvery, RTA=args.RTA)


# Show experiment duration
print(f"Run Time: {time.time()-start_time:0.4} seconds")


'''
** To start TensorBoard, run the following command in your terminal with your specific path to aerospacerl:**
Aircraft:
tensorboard --logdir aerospacerl/RL/runs/ac
Spacecraft:
tensorboard --logdir aerospacerl/RL/runs/sc
'''
