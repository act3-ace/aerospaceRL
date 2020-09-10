'''
2D Spacecraft Docking Environment

Created by Kyle Dunlap and Kai Delsing
Mentor: Kerianne Hobbs

Description:
	A deputy spacecraft is trying to dock with the chief spacecraft in Hill's frame

Observation (Deputy):
	Type: Box(4)
	Num 	Observation 		Min 	Max
	0		x position			-Inf 	+Inf
	1		y position			-Inf 	+Inf
	2		x velocity			-Inf 	+Inf
	3		y velocity			-Inf 	+Inf

Actions (Discrete or Continuous):
	Type: Discrete(9)
	Num 	X Force	 	Y Force	 (Newtons)
	0		-1			-1
	1		-1			 0
	2		-1			 1
	3		 0			-1
	4		 0			 0
	5		 0			 1
	6		 1			-1
	7		 1			 0
	8		 1			 1

	Type: Continuous Box(2,)
	[X Force, Y Force]
	Each value between -1 and +1 Newtons

Reward:
	+1 for successfully docking
	-1 for going out of bounds, running out of time/fuel
	small negative reward for getting closer to chief, larger negative reward for getting further
	small negative reward for going over max velocity constraint or if RTA is on
	small negative reward if below min velocity

Starting State:
	Deputy start 1000 m away from chief at random angle
	x and y velocity are both between -1.44 and +1.44 m/s

Episode Termination:
	Deputy docks with chief
	Deputy hits chief
	Deputy goes out of bounds
	Out of time
	Out of fuel

Integrators:
	Euler: Simplest and quickest
	Quad: ~4 times slower than Euler
	RK45: ~10 times slower than Quad

Run Time Assurance:
	Used to limit maximum velocity
'''

import gym
from gym import spaces
from gym.utils import seeding
import math
import random
import numpy as np
from fractions import Fraction
from scipy import integrate
from aero_gym.envs.rendering import DockingRender as render

class SpacecraftDocking(gym.Env):

	def __init__(self):
		self.x_chief = 0 # m
		self.y_chief = 0 # m
		self.position_deputy = 1000 # m (Relative distance from chief)
		self.mass_deputy = 12 # kg
		self.n = 0.001027 # rad/sec (mean motion)
		self.tau = 1 # sec (time step)
		self.integrator = 'Euler' # Either 'Quad', 'RK45', or 'Euler' (default)
		self.force_magnitude = 1 # Newtons
		self.x_threshold = 1.5 * self.position_deputy # m (In either direction)
		self.y_threshold = 1.5 * self.position_deputy # m (In either direction)
		self.pos_threshold = 0.1 # m (|x| and |y| must be less than this to dock)
		self.vel_threshold = 0.2 # m/s (Relative velocity must be less than this to dock)
		self.max_time = 4000 # seconds
		self.max_control = 2500 # Newtons
		self.init_velocity = (self.position_deputy + 625) / 1125 # m/s (+/- x and y)
		self.RTA_reward = 'NoRTA' # Changes reward for different RTA, either 'NoRTA', 'CBF', 'SVL', 'ASIF', or 'SBSF'

		#For Tensorboard Plots#
		self.RTA_on = False # Flag for if RTA is on or not, used for rewards
		self.success = 0 # Used to count success rate for an epoch
		self.failure = 0 # Used to count out of bounds rate for an epoch
		self.crash = 0 # Used to count crash rate for an epoch
		self.overtime = 0 # Used to count over max time/control for an epoch

		#Thrust & Particle Variables#
		self.thrustVis = 'Particle' #what type of thrust visualization to use. 'Particle', 'Block', 'None'
		self.particles = [] #list containing particle references
		self.p_obj = [] #list containing particle objects
		self.trans = [] #list containing particle
		self.p_velocity = 20 #velocity of particle
		self.p_ttl = 4 #(steps) time to live per particle
		self.p_var = 3 #(deg) the variation of launch angle (multiply by 2 to get full angle)

		#Ellipse Variables#
		self.ellipse_a1 = 200 #m
		self.ellipse_b1 = 100 #m
		self.ellipse_a2 = 40 #m
		self.ellipse_b2 = 20 #m
		self.ellipse_quality = 150 #1/x * pi

		#Trace Variables#
		self.trace = 5 #(steps)spacing between trace dots
		self.traceMin = True #sets trace size to 1 (minimum) if true
		self.tracectr = self.trace

		#Customization Options#
		self.viewer = None #gym thing - must be set to show up
		self.showRes = False #if set to true, it will print resolution
		self.scale_factor = .5 * 500 / self.position_deputy #sets the size of the rendering
		self.velocityArrow = True #if velocity arrow is shown
		self.forceArrow = True #if force arrow is shown
		self.bg_color = (0, 0, .15) #r,g,b
		self.stars = 200 #sets number of stars
		self.termination_condition = False # Set to true to print termination condition


		high = np.array([np.finfo(np.float32).max, # x position (Max possible value +inf)
			np.finfo(np.float32).max, # y position
			np.finfo(np.float32).max, # x velocity
			np.finfo(np.float32).max], # y velocity
			dtype=np.float32)

		self.action_select() # Select discrete or continuous action space

		if self.action_type == 'Discrete': # Discrete action space
			self.action_space = spaces.Discrete(9)
		else: # Continuous action space
			self.action_space = spaces.Box(np.array([-self.force_magnitude, -self.force_magnitude]), np.array([self.force_magnitude, self.force_magnitude]), dtype=np.float64)

		self.observation_space = spaces.Box(-high, high, dtype=np.float32) # Continuous observation space

		self.seed() # Generate random seed

		self.reset() # Reset environment when initialized

	def action_select(self): # Defines action type
		 self.action_type = 'Discrete'

	def seed(self, seed=None): # Sets random seed
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self): #called before each episode
		self.steps = -1 # Step counter
		self.control_input = 0 # Used to sum total control input for an episode

		if self.RTA_reward == 'CBF' or self.RTA_reward == 'ASIF' or self.RTA_reward == 'SBSF':
			self.max_time = 4000 # seconds

		# Use random angle to calculate x and y position
		theta = self.np_random.uniform(low=0, high=2*math.pi) # random angle, starts 10km away
		self.x_deputy = self.position_deputy*math.cos(theta) # m
		self.y_deputy = self.position_deputy*math.sin(theta) # m
		# Random x and y velocity
		x_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity) # m/s
		y_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity) # m/s

		self.rH = self.position_deputy # m (Relative distance from chief)

		self.state = np.array([self.x_deputy, self.y_deputy, x_dot, y_dot]) # Define observation state
		return self.state

	def step(self, action):
		self.steps += 1 # step counter

		if self.action_type == 'Discrete':
            # Stop program if invalid action is used
			assert self.action_space.contains(action), "Invalid action"
		else:
			# Clip action to be within boundaries - only for continuous
			action = np.clip(action, -self.force_magnitude, self.force_magnitude)

		# Extract current state data
		x, y, x_dot, y_dot = self.state
		rH_old = self.rH

		if self.action_type == 'Discrete': # Discrete action space
			if action == 0:
				self.x_force = -self.force_magnitude
				self.y_force = -self.force_magnitude
			elif action == 1:
				self.x_force = -self.force_magnitude
				self.y_force = 0
			elif action == 2:
				self.x_force = -self.force_magnitude
				self.y_force = self.force_magnitude
			elif action == 3:
				self.x_force = 0
				self.y_force = -self.force_magnitude
			elif action == 4:
				self.x_force = 0
				self.y_force = 0
			elif action == 5:
				self.x_force = 0
				self.y_force = self.force_magnitude
			elif action == 6:
				self.x_force = self.force_magnitude
				self.y_force = -self.force_magnitude
			elif action == 7:
				self.x_force = self.force_magnitude
				self.y_force = 0
			elif action == 8:
				self.x_force = self.force_magnitude
				self.y_force = self.force_magnitude

		else: # Continuous action space
			self.x_force, self.y_force = action

		# Add total force for given time period
		self.control_input += (abs(self.x_force) + abs(self.y_force)) * self.tau


		# Integrate Acceleration and Velocity
		if self.integrator == 'RK45': # Runge-Kutta Integrator
			# Define acceleration functions
			def x_acc_int(t,x):
				return (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + (self.x_force / self.mass_deputy)
			def y_acc_int(t,y):
				return (-2 * self.n * x_dot) + (self.y_force / self.mass_deputy)
			# Integrate acceleration to calculate velocity
			x_dot = integrate.solve_ivp(x_acc_int,(0,self.tau),[x_dot]).y[-1,-1]
			y_dot = integrate.solve_ivp(y_acc_int,(0,self.tau),[y_dot]).y[-1,-1]
			# Define velocity functions
			def vel_int(t,x):
				return x_dot, y_dot
			# Integrate velocity to calculate position
			xtemp, ytemp = integrate.solve_ivp(vel_int,(0,self.tau),[x,y]).y
			x = xtemp[-1]
			y = ytemp[-1]

		elif self.integrator == 'Euler': # Simple Euler Integrator
			# Define acceleration functions
			x_acc = (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + (self.x_force / self.mass_deputy)
			y_acc = (-2 * self.n * x_dot) + (self.y_force / self.mass_deputy)
			# Integrate acceleration to calculate velocity
			x_dot = x_dot + x_acc * self.tau
			y_dot = y_dot + y_acc * self.tau
			# Integrate velocity to calculate position
			x = x + x_dot * self.tau
			y = y + y_dot * self.tau

		else: # Default 'Quad' Integrator
			# Integrate acceleration to calculate velocity
			x_dot = x_dot + integrate.quad(lambda x:(3 * self.n ** 2 * x) + (2 * self.n * y_dot) + (self.x_force / self.mass_deputy), 0, self.tau)[0]
			y_dot = y_dot + integrate.quad(lambda y:(-2 * self.n * x_dot) + (self.y_force / self.mass_deputy), 0, self.tau)[0]
			# Integrate velocity to calculate position
			x = x + integrate.quad(lambda x:x_dot, 0, self.tau)[0]
			y = y + integrate.quad(lambda y:y_dot, 0, self.tau)[0]

		# Define new observation state
		self.state = np.array([x, y, x_dot, y_dot])

		# Done if any of these conditions are met:
		done = bool(
			(abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold)
			or abs(x) > self.x_threshold
			or abs(y) > self.y_threshold
			or self.control_input > self.max_control
			or self.steps * self.tau > self.max_time
		)

		self.rH = np.sqrt(x**2 + y**2) # Relative distance between deputy and chief (assume chief is always at origin)
		self.vH = np.sqrt(x_dot**2 + y_dot**2) # Velocity Magnitude
		self.vH_max = 2 * self.n * self.rH + self.vel_threshold # Max Velocity
		self.vH_min = 1/2 * self.n * self.rH - self.vel_threshold # Min Velocity


		# Rewards
		if not done:
			reward = (-1+rH_old-self.rH)/2000 * self.tau #min((-1+rH_old-self.rH)/2000, -0.00005) * self.tau  # Negative reward for getting closer/further

			if self.vH < self.vH_min:
				reward += -0.0075*abs(self.vH-self.vH_min) * self.tau # Negative reward for being below min velocity

			if self.RTA_reward == 'NoRTA' and self.vH > self.vH_max:
				reward += -0.0035*abs(self.vH-self.vH_max) * self.tau # Negative reward for being over max velocity

			if self.RTA_on:
				reward += -0.001 * self.tau # Negative reward if RTA is on

			if self.vH < 2*self.vel_threshold and (self.RTA_on or self.vH < self.vH_min or self.vH > self.vH_max):
				if self.RTA_reward == 'NoRTA':
					reward += -0.0075/2  * self.tau # Larger negative reward if violating constraint close to docking

				# elif self.RTA_reward == 'SVL' or self.RTA_reward == 'SBSF':
				# 	reward += -0.005/2  * self.tau # Larger negative reward if violating constraint close to docking
				# else: #ASIF Case -can possibly remove if ASIF is edited
				# 	if self.vH < 0.01:
				# 		reward += -0.001/2  * self.tau # Negative reward if velocity is zero

				else: # Try this for ASIF. If it doesn't work, go back to previous rewards
					reward += -0.005/2  * self.tau

		elif abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold:
			# if self.RTA_reward == 'NoRTA' or self.RTA_reward == 'SVL' or self.RTA_reward == 'SBSF': #this can be removed if ASIF fixed (but keep if else that is nested)
			# 	if self.vH > self.vel_threshold:
			# 		reward = -0.001 # Negative reward for crashing
			# 		self.crash += 1 # Track crash
			# 	else:
			# 		reward = 1 # +1 for docking
			# 		self.success += 1 # Track success
			# else: # can be removed if ASIF is fixed
			# 	reward = 1 # +1 for docking
			# 	self.success += 1 # Track success

			if self.vH > self.vel_threshold: # Try this for ASIF. If it doesn't work, go back to previous rewards
				reward = -0.001 # Negative reward for crashing
				self.crash += 1 # Track crash
			else:
				reward = 1 # +1 for docking
				self.success += 1 # Track success

		elif self.control_input > self.max_control or self.steps * self.tau > self.max_time:
			reward = -1 # -1 for over max time or control
			self.overtime += 1 # Track overtime

		else:
			reward = -1 # -1 for going out of bounds
			self.failure += 1 # Track failure


		# Print termination condition (if True)
		if done and self.termination_condition:
			if abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold:
				if self.vH < self.vel_threshold:
					print('Termination Condition: Successfully Docked')
				else:
					print('Termination Condition: Crash')
			elif x < -self.x_threshold:
				print('Termination Condition: BOUNDARY - neg x thresh')
			elif x > self.x_threshold:
				print('Termination Condition: BOUNDARY - pos x thresh')
			elif y < -self.y_threshold:
				 print('Termination Condition: BOUNDARY - neg y thresh')
			elif y > self.y_threshold:
				print('Termination Condition: BOUNDARY - pos y thresh')
			elif self.control_input > self.max_control:
				print('Termination Condition: Out of Fuel')
			elif self.steps * self.tau > self.max_time:
				print('Termination Condition: Out of Time')
			else:
				print('Unknown Termination Condition')

		# Return obs, rew, done, info
		return self.state, reward, done, {}

	# Used to check if velocity is over max velocity constraint
	def check_velocity(self, x_force, y_force):
		# Extract current state data
		x, y, x_dot, y_dot = self.state
		# Define acceleration functions
		x_acc = (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + (x_force / self.mass_deputy)
		y_acc = (-2 * self.n * x_dot) + (y_force / self.mass_deputy)
		# Integrate acceleration to calculate velocity
		x_dot = x_dot + x_acc * self.tau
		y_dot = y_dot + y_acc * self.tau

		# Check if over max velocity, and return True if it is violating constraint
		rH = np.sqrt(x**2 + y**2) # m, distance between deputy and chief
		vH = np.sqrt(x_dot**2 + y_dot**2) # Velocity Magnitude
		vH_max = 2 * self.n * self.rH + self.vel_threshold # Max Velocity # Max Velocity

		# If violating, return True
		if vH > vH_max:
			value = True
		else:
			value = False

		# Calculate velocity angle
		vtheta = math.atan(y_dot/x_dot)
		if x_dot <0:
			vtheta += math.pi

		return value, vH_max, vtheta

	# Run Time Assurance for discrete actions- based on velocity constraint
	def RTA(self, action):
		# Extract current state data
		x, y, x_dot, y_dot = self.state

		# Define force value for each possible action
		if action == 0:
			x_force = -self.force_magnitude
			y_force = -self.force_magnitude
		elif action == 1:
			x_force = -self.force_magnitude
			y_force = 0
		elif action == 2:
			x_force = -self.force_magnitude
			y_force = self.force_magnitude
		elif action == 3:
			x_force = 0
			y_force = -self.force_magnitude
		elif action == 4:
			x_force = 0
			y_force = 0
		elif action == 5:
			x_force = 0
			y_force = self.force_magnitude
		elif action == 6:
			x_force = self.force_magnitude
			y_force = -self.force_magnitude
		elif action == 7:
			x_force = self.force_magnitude
			y_force = 0
		elif action == 8:
			x_force = self.force_magnitude
			y_force = self.force_magnitude

		# Check if over max velocity constraint
		over_max_vel, _, _ = self.check_velocity(x_force, y_force)

		# If violating:
		if over_max_vel:
			# act_list is a list of possible actions that do not violate max velocity constraint
			action = []
			# Test all 9 actions (except for one already tested)
			for i in range(9):
				if i == action:
					continue
				if i == 0:
					x_force = -self.force_magnitude
					y_force = -self.force_magnitude
				elif i == 1:
					x_force = -self.force_magnitude
					y_force = 0
				elif i == 2:
					x_force = -self.force_magnitude
					y_force = self.force_magnitude
				elif i == 3:
					x_force = 0
					y_force = -self.force_magnitude
				elif i == 4:
					x_force = 0
					y_force = 0
				elif i == 5:
					x_force = 0
					y_force = self.force_magnitude
				elif i == 6:
					x_force = self.force_magnitude
					y_force = -self.force_magnitude
				elif i == 7:
					x_force = self.force_magnitude
					y_force = 0
				elif i == 8:
					x_force = self.force_magnitude
					y_force = self.force_magnitude

				# Check if each is over max velocity
				over_max_vel, _, _ = self.check_velocity(x_force, y_force)

				# If that action does not violate max velocity constraint, append it to lists
				if not over_max_vel:
					action.append(i)

			# Set RTA flag to True
			self.RTA_on = True

		# If it is not violating constraint
		else:
			self.RTA_on = False

		# If RTA is on, returns list of possible actions. If RTA is off, returns original action
		return action

	# Rendering Functions
	def render(self):
		render.renderSim(self)

	def close(self):
		render.close(self)

# Used to define 'spacecraft-docking-continuous-v0'
class SpacecraftDockingContinuous(SpacecraftDocking):
	def action_select(self): # Defines continuous action space
		 self.action_type = 'Continuous'
