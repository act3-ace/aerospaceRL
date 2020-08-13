'''
OpenAI SpinningUp's implementation of PPO, designed to work with the spacecraft
docking environment

Modified by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core
from logx import EpochLogger
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import dubins_gym
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'asif')) # Add file path to asif

# Used to track actual duration of experiment
starttime = time.time()
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# Assumes spacecraftdockingrl is in your home directory
PATH = os.path.expanduser("~") + "/spacecraftdockingrl/RL_algorithms"
if not os.path.isdir(PATH):
	print('PATH ISSUE - UPDATE YOUR PATH')
	exit()

class PPOBuffer:
	"""
	A buffer for storing trajectories experienced by a PPO agent interacting
	with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
	for calculating the advantages of state-action pairs.
	"""

	def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
		self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
		self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
		self.adv_buf = np.zeros(size, dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.ret_buf = np.zeros(size, dtype=np.float32)
		self.val_buf = np.zeros(size, dtype=np.float32)
		self.logp_buf = np.zeros(size, dtype=np.float32)
		self.gamma, self.lam = gamma, lam
		self.ptr, self.path_start_idx, self.max_size = 0, 0, size

	def store(self, obs, act, rew, val, logp):
		"""
		Append one timestep of agent-environment interaction to the buffer.
		"""
		assert self.ptr < self.max_size     # buffer has to have room so you can store
		self.obs_buf[self.ptr] = obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.val_buf[self.ptr] = val
		self.logp_buf[self.ptr] = logp
		self.ptr += 1

	def finish_path(self, last_val=0):
		"""
		Call this at the end of a trajectory, or when one gets cut off
		by an epoch ending. This looks back in the buffer to where the
		trajectory started, and uses rewards and value estimates from
		the whole trajectory to compute advantage estimates with GAE-Lambda,
		as well as compute the rewards-to-go for each state, to use as
		the targets for the value function.

		The "last_val" argument should be 0 if the trajectory ended
		because the agent reached a terminal state (died), and otherwise
		should be V(s_T), the value function estimated for the last state.
		This allows us to bootstrap the reward-to-go calculation to account
		for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
		"""

		path_slice = slice(self.path_start_idx, self.ptr)
		rews = np.append(self.rew_buf[path_slice], last_val)
		vals = np.append(self.val_buf[path_slice], last_val)

		# the next two lines implement GAE-Lambda advantage calculation
		deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
		self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

		# the next line computes rewards-to-go, to be targets for the value function
		self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

		self.path_start_idx = self.ptr

	def get(self):
		"""
		Call this at the end of an epoch to get all of the data from
		the buffer, with advantages appropriately normalized (shifted to have
		mean zero and std one). Also, resets some pointers in the buffer.
		"""
		assert self.ptr == self.max_size    # buffer has to be full before you can get
		self.ptr, self.path_start_idx = 0, 0
		# the next two lines implement the advantage normalization trick
		adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
		self.adv_buf = (self.adv_buf - adv_mean) / adv_std
		data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
					adv=self.adv_buf, logp=self.logp_buf)
		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
		steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
		vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=None,
		target_kl=0.01, logger_kwargs=dict(), save_freq=10, TensorBoard=True, save_nn=True,
		save_every=1000, load_latest=False, load_custom=False, LoadPath=None, plot=True, RTA=None):
	"""
	Proximal Policy Optimization (by clipping),

	with early stopping based on approximate KL

	Args:
		env_fn : A function which creates a copy of the environment.
			The environment must satisfy the OpenAI Gym API.

		actor_critic: The constructor method for a PyTorch Module with a
			``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
			module. The ``step`` method should accept a batch of observations
			and return:

			===========  ================  ======================================
			Symbol       Shape             Description
			===========  ================  ======================================
			``a``        (batch, act_dim)  | Numpy array of actions for each
										   | observation.
			``v``        (batch,)          | Numpy array of value estimates
										   | for the provided observations.
			``logp_a``   (batch,)          | Numpy array of log probs for the
										   | actions in ``a``.
			===========  ================  ======================================

			The ``act`` method behaves the same as ``step`` but only returns ``a``.

			The ``pi`` module's forward call should accept a batch of
			observations and optionally a batch of actions, and return:

			===========  ================  ======================================
			Symbol       Shape             Description
			===========  ================  ======================================
			``pi``       N/A               | Torch Distribution object, containing
										   | a batch of distributions describing
										   | the policy for the provided observations.
			``logp_a``   (batch,)          | Optional (only returned if batch of
										   | actions is given). Tensor containing
										   | the log probability, according to
										   | the policy, of the provided actions.
										   | If actions not given, will contain
										   | ``None``.
			===========  ================  ======================================

			The ``v`` module's forward call should accept a batch of observations
			and return:

			===========  ================  ======================================
			Symbol       Shape             Description
			===========  ================  ======================================
			``v``        (batch,)          | Tensor containing the value estimates
										   | for the provided observations. (Critical:
										   | make sure to flatten this!)
			===========  ================  ======================================


		ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
			you provided to PPO.

		seed (int): Seed for random number generators.

		steps_per_epoch (int): Number of steps of interaction (state-action pairs)
			for the agent and the environment in each epoch.

		epochs (int): Number of epochs of interaction (equivalent to
			number of policy updates) to perform.

		gamma (float): Discount factor. (Always between 0 and 1.)

		clip_ratio (float): Hyperparameter for clipping in the policy objective.
			Roughly: how far can the new policy go from the old policy while
			still profiting (improving the objective function)? The new policy
			can still go farther than the clip_ratio says, but it doesn't help
			on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
			denoted by :math:`\epsilon`.

		pi_lr (float): Learning rate for policy optimizer.

		vf_lr (float): Learning rate for value function optimizer.

		train_pi_iters (int): Maximum number of gradient descent steps to take
			on policy loss per epoch. (Early stopping may cause optimizer
			to take fewer than this.)

		train_v_iters (int): Number of gradient descent steps to take on
			value function per epoch.

		lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
			close to 1.)

		max_ep_len (int): Maximum length of trajectory / episode / rollout.

		target_kl (float): Roughly what KL divergence we think is appropriate
			between new and old policies after an update. This will get used
			for early stopping. (Usually small, 0.01 or 0.05.)

		logger_kwargs (dict): Keyword args for EpochLogger.

		save_freq (int): How often (in terms of gap between epochs) to save
			the current policy and value function.

		TensorBoard (bool): True plots to TensorBoard, False does not

		save_nn (bool): True saves neural network data, False does not

		save_every (int): How often to save neural network

		load_latest (bool): Load last saved neural network data before training

		load_custom (bool): Load custom neural network data file before training

		LoadPath (str): Path for custom neural network data file

		plot (bool): Plots trajectories and velocity if True, no plot if False

		RTA (str): RTA framework, either 'CBF', 'Velocity', 'IASIF', or
			'ISimplex'

	"""

	# Special function to avoid certain slowdowns from PyTorch + MPI combo.
	setup_pytorch_for_mpi()

	# Set up logger and save configuration
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	# Instantiate environment
	env = env_fn()
	obs_dim = env.observation_space.shape
	act_dim = env.action_space.shape

	# Random seed for each cpu
	seed += 1 * proc_id()
	env.seed(seed)

	# Create actor-critic module
	ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

	# Load model if True
	if load_latest:
		models = glob.glob(f"{PATH}/models/sc/*")
		LoadPath = max(models, key=os.path.getctime)
		ac.load_state_dict(torch.load(LoadPath))
	elif load_custom:
		ac.load_state_dict(torch.load(LoadPath))

	# Sync params across processes
	sync_params(ac)

	# Count variables
	var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
	logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

	# Set up experience buffer
	local_steps_per_epoch = int(steps_per_epoch / num_procs())
	buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

	# Set up function for computing PPO policy loss
	def compute_loss_pi(data):
		obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

		# Policy loss
		pi, logp = ac.pi(obs, act)
		ratio = torch.exp(logp - logp_old)
		clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
		loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

		# Useful extra info
		approx_kl = (logp_old - logp).mean().item()
		ent = pi.entropy().mean().item()
		clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
		clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
		pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

		return loss_pi, pi_info

	# Set up function for computing value loss
	def compute_loss_v(data):
		obs, ret = data['obs'], data['ret']
		return ((ac.v(obs) - ret)**2).mean()

	# Set up optimizers for policy and value function
	pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
	vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

	# Set up model saving
	logger.setup_pytorch_saver(ac)

	def update():
		data = buf.get()

		pi_l_old, pi_info_old = compute_loss_pi(data)
		pi_l_old = pi_l_old.item()
		v_l_old = compute_loss_v(data).item()

		# Train policy with multiple steps of gradient descent
		for i in range(train_pi_iters):
			pi_optimizer.zero_grad()
			loss_pi, pi_info = compute_loss_pi(data)
			kl = mpi_avg(pi_info['kl'])
			if kl > 1.5 * target_kl:
				logger.log('Early stopping at step %d due to reaching max kl.'%i)
				break
			loss_pi.backward()
			mpi_avg_grads(ac.pi)    # average grads across MPI processes
			pi_optimizer.step()

		logger.store(StopIter=i)

		# Value function learning
		for i in range(train_v_iters):
			vf_optimizer.zero_grad()
			loss_v = compute_loss_v(data)
			loss_v.backward()
			mpi_avg_grads(ac.v)    # average grads across MPI processes
			vf_optimizer.step()

		# Log changes from update
		kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
		logger.store(LossPi=pi_l_old, LossV=v_l_old,
					 KL=kl, Entropy=ent, ClipFrac=cf,
					 DeltaLossPi=(loss_pi.item() - pi_l_old),
					 DeltaLossV=(loss_v.item() - v_l_old))

	# Use best possible action for plots
	def get_best_action(obs):
		with torch.no_grad():
			act = ac.pi.mu_net(obs).numpy()
		return act

	# Import ASIF
	asif_on = False
	if RTA == 'CBF':
		from CBF_for_speed_limit import ASIF
		asif_on = True
	elif RTA == 'Velocity':
		from Simple_velocity_limit import ASIF
		asif_on = True
	elif RTA == 'IASIF':
		from IASIF import ASIF
		asif_on = True
	elif RTA == 'ISimplex':
		from ISimplex import ASIF
		asif_on = True

	# Call ASIF, define action conversion
	if asif_on:
		env.RTA_reward = RTA

		asif = ASIF(env)

		def asif_act(obs, act):
			act = np.clip(act, -env.force_magnitude, env.force_magnitude)
			x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
			u_des = np.array([[act[0]], [act[1]], [0]])
			u = asif.main(x0, u_des)
			new_act = [u[0,0], u[1,0]]
			if abs(np.sqrt(new_act[0]**2+new_act[1]**2) - np.sqrt(act[0]**2+act[1]**2)) < 0.0001:
				env.RTA_on = False
			else:
				env.RTA_on = True
			return new_act

	# Prepare for interaction with environment
	start_time = time.time()
	o, ep_ret, ep_len = env.reset(), 0, 0
	total_episodes = 0
	RTA_percent = 0

	# Create TensorBoard file if True
	if TensorBoard and proc_id() == 0:
		Name = f"{PATH}/runs/sc/Spacecraft-docking-" + current_time
		writer = SummaryWriter(Name)

	# Set up plot if true
	if plot and proc_id() == 0:
		fig, (ax1,ax2) = plt.subplots(1,2)

	# Main loop: collect experience in env and update/log each epoch
	for epoch in range(epochs):
		batch_ret = [] # Track episode returns
		batch_len = [] # Track episode lengths
		batch_RTA_percent = [] # Track precentage of time RTA is on
		env.success = 0 # Track episode success rate
		env.failure = 0 # Track episode failure rate
		env.crash = 0 # Track episode crash rate
		env.overtime = 0 # Track episode over max time/control rate
		episodes = 0 # Track episodes
		delta_v = [] # Track episode total delta v
		for t in range(local_steps_per_epoch):
			a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
			if asif_on: # If RTA is on, get RTA action
				RTA_a = asif_act(o, a)
				if env.RTA_on:
					RTA_percent += 1
				next_o, r, d, _ = env.step(RTA_a)
			else: # If RTA is off, pass through desired action
				next_o, r, d, _ = env.step(a)
				over_max_vel, _, _ = env.check_velocity(a[0], a[1])
				if over_max_vel:
					RTA_percent += 1
			ep_ret += r
			ep_len += 1

			# save and log
			buf.store(o, a, r, v, logp)
			logger.store(VVals=v)

			# Update obs (critical!)
			o = next_o

			timeout = ep_len == max_ep_len
			terminal = d or timeout
			epoch_ended = t==local_steps_per_epoch-1

			if terminal or epoch_ended:
				if epoch_ended and not(terminal):
					print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
				# if trajectory didn't reach terminal state, bootstrap value target
				if timeout or epoch_ended:
					_, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
				else:
					v = 0
				buf.finish_path(v)
				if terminal:
					# only save EpRet / EpLen if trajectory finished
					logger.store(EpRet=ep_ret, EpLen=ep_len)
					batch_ret.append(ep_ret)
					batch_len.append(ep_len)
					episodes += 1
					delta_v.append(env.control_input/env.mass_deputy)
				batch_RTA_percent.append(RTA_percent/ep_len*100)
				RTA_percent = 0
				o, ep_ret, ep_len = env.reset(), 0, 0


		total_episodes += episodes
		# Track success, failure, crash, overtime rates
		if episodes != 0:
			success_rate = env.success/episodes
			failure_rate = env.failure/episodes
			crash_rate = env.crash/episodes
			overtime_rate = env.overtime/episodes
		else:
			success_rate = 0
			failure_rate = 0
			crash_rate = 0
			overtime_rate = 0

		# Save model
		if (epoch % save_freq == 0) or (epoch == epochs-1):
			logger.save_state({'env': env}, None)

		# Perform PPO update!
		update()

		# Log info about epoch
		logger.log_tabular('Epoch', epoch)
		logger.log_tabular('EpRet', with_min_and_max=True)
		logger.log_tabular('EpLen', average_only=True)
		logger.log_tabular('VVals', with_min_and_max=True)
		logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
		logger.log_tabular('LossPi', average_only=True)
		logger.log_tabular('LossV', average_only=True)
		logger.log_tabular('DeltaLossPi', average_only=True)
		logger.log_tabular('DeltaLossV', average_only=True)
		logger.log_tabular('Entropy', average_only=True)
		logger.log_tabular('KL', average_only=True)
		logger.log_tabular('ClipFrac', average_only=True)
		logger.log_tabular('StopIter', average_only=True)
		logger.log_tabular('Time', time.time()-start_time)
		logger.dump_tabular()

		# Average data over all cpus
		avg_batch_ret = mpi_avg(np.mean(batch_ret))
		avg_batch_len = mpi_avg(np.mean(batch_len))
		avg_success_rate = mpi_avg(success_rate)
		avg_failure_rate = mpi_avg(failure_rate)
		avg_crash_rate = mpi_avg(crash_rate)
		avg_overtime_rate = mpi_avg(overtime_rate)
		avg_delta_v = mpi_avg(np.mean(delta_v))
		avg_RTA_percent = mpi_avg(np.mean(batch_RTA_percent))

		if proc_id() == 0: # Only on one cpu
			# Plot 5 epochs over entire range, run new episode using best case actions
			if plot and (epoch == epochs - 1 or (epoch % math.ceil(epochs/5) == 0 and epoch != 0)):
				x = [] # Track x trajectory
				y = [] # Track y trajectory
				if epoch == epochs - 1:
					vH = [] # Track velocity
					if RTA == 'CBF':
						cbf = [] # Track max velocity
					elif RTA == 'IASIF' or RTA == 'ISimplex':
						iasif = [] # Track max velocity
					else:
						vH_max = [] # Track max velocity
				# Reset variables
				o = env.reset()
				d = False
				while not d:
					a = get_best_action(torch.as_tensor(o, dtype=torch.float32))
					if asif_on:
						a = asif_act(o, a)
					o, _, d, _ = env.step(a)
					x.append(o[0])
					y.append(o[1])
					if epoch == epochs - 1:
						vH.append(env.vH)
						if RTA == 'CBF':
							cbf.append(np.sqrt(asif.K * env.rH ** 2))
						elif RTA == 'IASIF' or RTA == 'ISimplex':
							iasif.append(asif.K1_s * env.rH + asif.K2_s)
						else:
							vH_max.append(env.vH_max)

				if epoch == epochs - 1:
					# Plot solid line for last epoch
					ax1.plot(x,y, label=epoch, linestyle='-')
					# Plot velocity
					ax2.plot(range(len(vH)),vH, label='Velocity')
					if RTA == 'CBF':
						ax2.plot(range(len(cbf)), cbf, 'r', label='CBF Max Velocity')
					elif RTA == 'IASIF':
						ax2.plot(range(len(iasif)), iasif, 'r', label='IASIF Max Velocity')
					elif RTA == 'ISimplex':
						ax2.plot(range(len(iasif)), iasif, 'r', label='ISimplex Max Velocity')
					else:
						ax2.plot(range(len(vH_max)),vH_max, 'r', label='Max Velocity')
				else:
					# Dashed lines for all other epochs
					ax1.plot(x,y, label=epoch, linestyle='--')

			# Plot to TensorBoard if True, only on one cpu
			if TensorBoard:
				writer.add_scalar('Return', avg_batch_ret, epoch)
				writer.add_scalar('Episode-Length', avg_batch_len*env.tau, epoch)
				writer.add_scalar('Success-Rate', avg_success_rate*100, epoch)
				writer.add_scalar('Failure-Rate', avg_failure_rate*100, epoch)
				writer.add_scalar('Crash-Rate', avg_crash_rate*100, epoch)
				writer.add_scalar('Overtime-Rate', avg_overtime_rate*100, epoch)
				writer.add_scalar('Delta-V', avg_delta_v, epoch)
				writer.add_scalar('RTA-on-percent', avg_RTA_percent, epoch)

			# Save neural network if true, can change to desired location
			if save_nn and epoch % save_every == 0 and epoch != 0:
				if not os.path.isdir(f"{PATH}/models"):
					os.mkdir(f"{PATH}/models")
				if not os.path.isdir(f"{PATH}/models/sc"):
					os.mkdir(f"{PATH}/models/sc")
				Name2 = f"{PATH}/models/sc/Spacecraft-docking-" + current_time + f"-epoch{epoch}.dat"
				torch.save(ac.state_dict(), Name2)

	# Average episodes per hour, episode per epoch
	ep_hr = mpi_avg(total_episodes)*args.cpu/(time.time()-start_time)*3600
	ep_Ep = mpi_avg(total_episodes)*args.cpu/(epoch+1)

	# Plot on one cpu
	if proc_id() == 0:
		if plot:
			# Plot chief
			ax1.plot(0,0,'kx',markersize=8)
			# Plot boundaries
			ax1.vlines(env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
			ax1.vlines(-env.x_threshold,-env.y_threshold,env.y_threshold,colors='r')
			ax1.hlines(-env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
			ax1.hlines(env.y_threshold,-env.x_threshold,env.x_threshold,colors='r')
			ax1.set_xlabel('X [m]')
			ax1.set_ylabel('Y [m]')
			ax1.grid(True)
			ax1.legend()
			ax1.set_title('Trajectories')
			ax2.set_xlabel('Time Steps')
			ax2.set_ylabel('Velocity (m/s)')
			ax2.grid(True)
			ax2.legend()
			ax2.set_title('Velocity Over Time for Last Episode')

		# Save neural network
		if save_nn:
			if not os.path.isdir(f"{PATH}/models"):
				os.mkdir(f"{PATH}/models")
			if not os.path.isdir(f"{PATH}/models/sc"):
				os.mkdir(f"{PATH}/models/sc")
			Name2 = f"{PATH}/models/sc/Spacecraft-docking-" + current_time + "-final.dat"
			torch.save(ac.state_dict(), Name2)

		# Print statistics on episodes
		print(f"Episodes per hour: {ep_hr:.0f}, Episodes per epoch: {ep_Ep:.0f}, Epochs per hour: {(epoch+1)/(time.time()-start_time)*3600:.0f}")

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='spacecraft-docking-continuous-v0') # Environment Name
	parser.add_argument('--hid', type=int, default=64) # Hidden layer nodes
	parser.add_argument('--l', type=int, default=2) # Number of hidden layers
	parser.add_argument('--gamma', type=float, default=0.99) # Discount factor
	parser.add_argument('--seed', '-s', type=int, default=0) # Seed for randomization
	parser.add_argument('--cpu', type=int, default=len(os.sched_getaffinity(0))) # Number of CPU cores (default is use all available)
	parser.add_argument('--steps', type=int, default=30000) # Steps per epoch (Defaults to enough to run at least one episode per core)
	parser.add_argument('--epochs', type=int, default=10000) # Number of epochs
	parser.add_argument('--exp_name', type=str, default='ppo') # Algorithm name (for logger)
	parser.add_argument('--NoTB', default=True, action='store_false') # Log to TnesorBoard - Add arg '--NoTB' if you don't want to log to TensorBoard
	parser.add_argument('--NoSave', default=True, action='store_false') # Save NN - Add arg '--NoSave' if you don't want to save NN
	parser.add_argument('--SaveEvery', type=int, default=500) # Save NN every _ epochs
	parser.add_argument('--LoadLatest', default=False, action='store_true') # Load NN - Add arg '--LoadLatest' to load last saved model
	parser.add_argument('--LoadCustom', default=False, action='store_true') # Load NN - Add arg '--LoadCustom' to load previous model (update path below)
	parser.add_argument('--NoPlot', default=True, action='store_false') # Plot - Add arg '--NoPlot' if you don't want to plot results
	parser.add_argument('--RTA', type=str, default='off') # Run Time Assurance - 4 options: 'CBF', 'Velocity', 'IASIF', or 'ISimplex'
	args = parser.parse_args()

	mpi_fork(args.cpu)  # run parallel code with mpi

	from run_utils import setup_logger_kwargs
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

	# For Custom Load Path:
	LoadPath = "spacecraftdockingrl/RL_algorithms/saved_models/PPObaseline2.dat"

	env_name = args.env
	ppo(lambda : gym.make(env_name), actor_critic=core.MLPActorCritic,
		ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
		seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
		logger_kwargs=logger_kwargs, TensorBoard=args.NoTB, save_nn=args.NoSave,
		save_every=args.SaveEvery, load_latest=args.LoadLatest, load_custom=args.LoadCustom,
		LoadPath = LoadPath, plot=args.NoPlot, RTA=args.RTA)

	# Show experiment duration
	if proc_id() == 0:
		print(f"Run Time: {time.time()-starttime:0.4} seconds")

	# Show plot if True
	if args.NoPlot and proc_id() == 0:
		plt.show()



#** To start TensorBoard, run the following command in your terminal with your specific path to spacecraftdockingrl:**
# Spacecraft:
# tensorboard --logdir spacecraftdockingrl/RL_algorithms/runs/sc
