'''
2D Dubins Aircraft Environment

Created by Kyle Dunlap and Kai Delsing
Mentor: Kerianne Hobbs

Description:
	An aircraft is trying to reach a goal while avoiding a mountain.  The environment
	is observed from above where the aircraft is moving right towards the goal.

Observation:
	Type: Box(8)
	Num 	Observation 		Min 	Max
	0		x position			-Inf 	+Inf
	1		y position			-Inf 	+Inf
	2		angle (from x-axis)	-Inf 	+Inf
	3		x velocity			-Inf 	+Inf
	4		y velocity			-Inf 	+Inf
	5		angular velocity	-Inf 	+Inf
	6		distance from mtn.	-Inf 	+Inf
	7		Angle wrt. mtn.		-Inf 	+Inf

Actions (Discrete or Continuous):
	Type: Discrete(5)
	Num 	Action (Rotate at:)
	0		-3 degrees/second
	1		-1.5 degrees/second
	2		0 degrees/second
	3		1.5 degrees/second
	4		3 degrees/second

	Type: Continuous Box(1,)
	One value, between -3 amd +3 degrees/second

Reward:
	-2 for every second until simulation is over
	-1000 for hitting mountain
	+1000 for reaching goal
	negative reward between 0 and -50 for coming within 500 ft of mountain
	-5000 for exiting screen

Starting State:
	Default:
	Start at origin (x=0,y=0), moving at 800 ft/sec in x-direction
	Mountain is at (45,000,0) ft and has radius 500 ft
	Goal is at x = 90,000 ft

	Random Aircraft/Mountain:
	Aircraft velocity and y position can be randomized
	Mountain position can be randomized
	Aircraft always points towards mountain

Episode Termination:
	Aircraft reaches goal
	Aircraft hits mountain
	Aircraft flies outside of thresholds
'''

# Import Libraries
import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
from fractions import Fraction
from scipy import integrate
from gym.envs.classic_control import rendering

class DubinsAircraft(gym.Env):

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 50
	}

	def __init__(self):
		self.x_goal = 90_000 #ft
		self.x_threshold = 5_000 #ft (To the left)  [-5,000 90,000]
		self.y_threshold = 25_000 #ft (Up or down) [-25,000 25,000]
		self.r_aircraft = 25 #ft - radius of the aircraft
		self.r_mountain = 500 #ft
		self.action_magnitude = 1.5 # degrees/sec
		self.tau = 0.1 #sec (time step)
		self.random_velocity = False # Set to True to allow random inital velocity
		self.random_mountain = False # Set to True to allow random inital position (aircraft always points at mountain)
		self.integrator = 'Euler' # Either 'Quad', 'RK45', or 'Euler' (default)
		self.success = 0

		self.scale_factor = 100  #inverse dialation of simulation size
		self.planescale = 1 #dialation of aircraft size
		self.ring = True  # add ring around aircraft for visual aid
		self.viewer = None # Initalize rendering window
		self.trace = 40 #every self.trace steps, trace dot is rendered. If it is 0, trace is disabled
		self.tracectr = self.trace #used to calculate trace spacing
		self.showRes = False #displays window resolution once
		self.termination_condition = False # Prints cause of termination when True

		high = np.array([np.finfo(np.float32).max, # x
			np.finfo(np.float32).max, # y
			np.finfo(np.float32).max, # theta
			np.finfo(np.float32).max, # x dot
			np.finfo(np.float32).max, # y dot
			np.finfo(np.float32).max, # theta dot
			np.finfo(np.float32).max, # mountain distance
			np.finfo(np.float32).max], # mountain angle
			dtype=np.float32)

		self.action_select() # Select discrete or continuous action space

		# Action space
		if self.action_type == 'Discrete':
			self.action_space = spaces.Discrete(5) # 5 discrete actions
		else: # Continuous action space
			self.action_space = spaces.Box(np.array([-2*self.action_magnitude]), np.array([2*self.action_magnitude]), dtype=np.float64)

		self.observation_space = spaces.Box(-high, high, dtype=np.float32) # Continuous observation space

		self.seed() # Generate random seed

		self.reset() # Reset environment when initialized

	def action_select(self):
		 self.action_type = 'Discrete'

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.control_input = 0 # Used to sum total control input for an episode
		
		if self.random_velocity:
			# Generate random value for aircraft velocity
			self.velocity = self.np_random.uniform(low=60, high=1145) #ft/s
		else:
			# Default values
			self.velocity = 800 #ft/s

		if self.random_mountain:
			# Generate random values for mountain x and y position and aircraft y position
			self.x_mountain = self.np_random.uniform(low=10000, high=self.x_goal-10000) #ft
			self.y_mountain = self.np_random.uniform(low=-self.y_threshold/2, high=self.y_threshold/2) #ft
			y = self.np_random.uniform(low=-self.y_threshold/2, high=self.y_threshold/2) #ft
			theta = math.atan((y-self.y_mountain)/(0-self.x_mountain)) #radians
		else:
			# Default values
			self.x_mountain = 45_000 #ft
			self.y_mountain = 0 #ft
			y = 0 #ft
			theta = 0 #radians

		# Calculate velocities
		x_dot = self.velocity * math.cos(theta)
		y_dot = self.velocity * math.sin(theta)

		# Calculate distance/angle wrt. mountain
		mountain_dist = np.sqrt(abs(0-self.x_mountain) ** 2 + abs(y-self.y_mountain) ** 2) - self.r_mountain - self.r_aircraft
		mountain_angle = math.atan((y-self.y_mountain)/(0-self.x_mountain))

		# Return observation states
		self.state = np.array([0, y, theta, x_dot, y_dot, 0, mountain_dist, mountain_angle])
		return self.state

	def step(self, action):
		if self.action_type == 'Discrete':
            # Stop program if invalid action is used
			assert self.action_space.contains(action), "Invalid action"
		else:
			action = np.clip(action, -2*self.action_magnitude, 2*self.action_magnitude)

		# Extract current state data
		x, y, theta, x_dot, y_dot, theta_dot, _, _ = self.state

		if self.action_type == 'Discrete': # Discrete action space (Default)
			self.action_magnitude_radians = self.action_magnitude * math.pi / 180
			# Take discrete action
			if action == 0:
				theta_dot = -2 * self.action_magnitude_radians
			elif action == 1:
				theta_dot = -1 * self.action_magnitude_radians
			elif action == 2:
				theta_dot = 0
			elif action == 3:
				theta_dot = 1 * self.action_magnitude_radians
			elif action == 4:
				theta_dot = 2 * self.action_magnitude_radians

		else: # Continuous action space
			theta_dot = action[0] * math.pi / 180

		if self.integrator == 'RK45': # Runge-Kutta integrator (slowest)
			# Integrate to calculate theta
			def theta_int(t,theta):
				return theta_dot
			theta = integrate.solve_ivp(theta_int,(0,self.tau),[theta]).y[-1,-1]

			# Calculate velocities
			x_dot = self.velocity * math.cos(theta)
			y_dot = self.velocity * math.sin(theta)

			# Integrate to calculate x and y
			def xy_int(t,x):
				return x_dot, y_dot
			xtemp, ytemp = integrate.solve_ivp(xy_int,(0,self.tau),[x,y]).y
			x = xtemp[-1]
			y = ytemp[-1]

		elif self.integrator == 'Quad': # Quad integrator (slower)
			# Integrate to calculate theta
			theta = theta + integrate.quad(lambda x:theta_dot, 0, self.tau)[0]

			# Calculate velocities
			x_dot = self.velocity * math.cos(theta)
			y_dot = self.velocity * math.sin(theta)

			# Integrate to calculate x and y
			x = x + integrate.quad(lambda x:x_dot, 0, self.tau)[0]
			y = y + integrate.quad(lambda x:y_dot, 0, self.tau)[0]

		else: # Default Euler Integrator (fastest)
			# Integrate to calculate theta
			theta = theta + theta_dot * self.tau

			# Calculate velocities
			x_dot = self.velocity * math.cos(theta)
			y_dot = self.velocity * math.sin(theta)

			# Integrate to calculate x and y
			x = x + x_dot * self.tau
			y = y + y_dot * self.tau


		# Calculate distance/angle wrt. mountain
		mountain_dist = np.sqrt(abs(x-self.x_mountain) ** 2 + abs(y-self.y_mountain) ** 2) - self.r_mountain - self.r_aircraft
		mountain_angle = math.atan((y-self.y_mountain)/(x-self.x_mountain))

		# Define new observation state
		self.state = np.array([x, y, theta, x_dot, y_dot, theta_dot, mountain_dist, mountain_angle])

		# Done if any of these conditions are met:
		done = bool(
			x >= self.x_goal
			or mountain_dist <= 0
			or x < -1 * self.x_threshold
			or y < -1 * self.y_threshold
			or y > self.y_threshold
		)

		# Rewards for each possible outcome
		if not done:
			reward = -2 * self.tau # -2 fpr every simulated second
			if mountain_dist <= 500:
				reward += (mountain_dist - 500)/ 10 # Decreasing reward for getting close to mountain
		elif mountain_dist <= 0:
			reward = -1000 # -1000 for crashing
		elif x >= self.x_goal:
			reward = 1000 # +1000 for reaching goal
			self.success += 1
		else:
			reward = -5000 # -5000 for going out of bounds

		# Print termination condition (if allowed)
		if done and self.termination_condition:
			if x >= self.x_goal:
				print('Termination Condition: GOAL')
			if mountain_dist <= 0:
				print('Termination Condition: CRASH - Mountain')
			elif x < -1 * self.x_threshold:
				print('Termination Condition: BOUNDARY - x thresh')
			elif y < -1 * self.y_threshold:
				 print('Termination Condition: BOUNDARY - neg y thresh')
			elif y > self.y_threshold:
				print('Termination Condition: BOUNDARY - pos y thresh')


		return self.state, reward, done, {}

	def render(self, mode='human'):
		x_thresh = self.x_threshold / self.scale_factor
		goal = self.x_goal / self.scale_factor
		y_thresh = self.y_threshold / self.scale_factor


		screen_width = x_thresh + goal + x_thresh  #calculate the screen width by adding the distance to the goal and the left threshold.
			#An extra x_threshold is added to provide buffer space
		screen_height = y_thresh * 2  #calculate the screen height by doubling the y thresh (up and down)
		screen_width, screen_height = int(screen_width), int(screen_height)  #convert the screen width and height to integers
		if self.showRes:
			print("Height: " + str(screen_height))
			print("Width: " + str(screen_width))
			self.showRes = False

		wingwidth = 25 * self.r_aircraft * self.planescale / self.scale_factor
		wingheight = 5 * self.r_aircraft * self.planescale / self.scale_factor
		bodywidth = 5 * self.r_aircraft * self.planescale / self.scale_factor
		bodyheight = 20 * self.r_aircraft * self.planescale / self.scale_factor
		tailwidth= 10 * self.r_aircraft * self.planescale / self.scale_factor
		if self.viewer is None:  #if no self.viewer exists, create it
			self.viewer = rendering.Viewer(screen_width, screen_height)  #creates a render

			b, t, l, r = 0, self.y_threshold * 2, 0, self.x_threshold * 2 + self.x_goal  #creates body dimensions
			sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates body polygon
			self.skytrans = rendering.Transform()  #allows body to be moved
			sky.add_attr(self.skytrans)
			sky.set_color(.7, .7, .9)  #sets color of body
			self.viewer.add_geom(sky)  #adds body to viewer

			b, t, l, r = -bodywidth/2, bodywidth/2, bodyheight/2, -bodyheight/2  #creates body dimensions
			body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates body polygon
			self.bodytrans = rendering.Transform()  #allows body to be moved
			body.add_attr(self.bodytrans)
			body.set_color(.2, .2, .2)  #sets color of body
			self.viewer.add_geom(body)  #adds body to viewer

			b, t, l, r = -wingwidth/2, wingwidth/2, wingheight/2, -wingheight/2  #creates wing dimensions
			wing = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates wing polygon
			self.wingtrans = rendering.Transform()  #allows wing to be moved
			wing.add_attr(self.wingtrans)
			wing.add_attr(self.bodytrans)  #sets wing as part of body
			wing.set_color(.3, .5, .3)  #sets color of wing
			self.viewer.add_geom(wing)  #adds wing to viewer

			b, t, l, r = -tailwidth/2, tailwidth/2, wingheight/2, -wingheight/2  #creates tail dimensions
			tail = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates tail polygon
			self.tailtrans = rendering.Transform(translation=(0, -bodyheight/3))  #translates the tail to the end of the body
			tail.add_attr(self.tailtrans)
			tail.set_color(.3, .3, .5)  #sets color of tail
			self.viewer.add_geom(tail)  #adds tail to render

			r_mtn = self.r_mountain / self.scale_factor
			mtn = rendering.make_circle(r_mtn)  #
			self.mtntrans = rendering.Transform()  #allows the mountain to be moved
			mtn.add_attr(self.mtntrans)
			mtn.set_color(.5, .5, .5)  #sets color of mountain
			self.viewer.add_geom(mtn)  #adds mountain into render

			goalLine = rendering.Line((goal, 2 * y_thresh),(goal, 0))  #creates goal line and endpoints
			self.goalLinetrans = rendering.Transform()  #allows goalLine to be moved
			goalLine.add_attr(self.goalLinetrans)
			goalLine.set_color(.9, .1, .1)  #sets color of goalLine
			self.viewer.add_geom(goalLine)  #adds goalLine into render

			if self.ring:
				ring = rendering.make_circle(wingwidth/2, 30, False)  #creates ring dimensions
				self.ringtrans = rendering.Transform()  #allows ring to be moved
				ring.add_attr(self.ringtrans)
				ring.add_attr(self.bodytrans)  #sets ring as part of body
				ring.set_color(.9, .0, .0)  #sets color of ring
				self.viewer.add_geom(ring)  #adds ring into render

		if self.state is None:  #if there is no state (either the simulation has not begun or it has ended), end
			print('No state')
			return None


		if self.trace != 0:
			if self.tracectr == self.trace:
				tracewidth = int(bodywidth / 2)
				if tracewidth < 1:
					tracewidth = 1
				trace = rendering.make_circle(tracewidth)  #creates trace dot
				self.tracetrans = rendering.Transform()  #allows trace to be moved
				trace.add_attr(self.tracetrans)
				trace.set_color(.9, .1, .9)  #sets color of trace
				self.viewer.add_geom(trace)  #adds trace into render
				self.tracectr = 0
			else:
				self.tracectr += 1

		x = self.state
		tx, ty = x[0] / self.scale_factor, (x[1] + self.y_threshold) / self.scale_factor  #pulls the state of the x and y coordinates
		self.bodytrans.set_rotation(x[2])  #rotate body
		self.bodytrans.set_translation(tx, ty)  #translate body

		self.tracetrans.set_translation(tx, ty)  #translate trace

		x_mtn = self.x_mountain / self.scale_factor
		y_mtn = (self.y_mountain + self.y_threshold) / self.scale_factor
		#self.mtntrans.set_rotation(x[])  #rotate mountain
		self.mtntrans.set_translation(x_mtn, y_mtn)  #translate mountain

		d=-bodyheight/3  #set distance  #find distance to travel
		self.tailtrans.set_rotation(x[2])  #rotate tail
		thetashift = x[2]-90.0  #convert graphics direction to Cartesian angles
		radtheta = (thetashift * 3.1415926535) / 180.0  #convert angle to radians
		transx, transy = math.sin(radtheta) * d, math.cos(radtheta) * d  #use trig to find actual x and y translations
		self.tailtrans.set_translation(tx - transx, ty + transy)  #translate tail

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):  #if a viewer exists, close and kill it
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None


class DubinsAircraftContinuous(DubinsAircraft):
	def action_select(self):
		 self.action_type = 'Continuous'
