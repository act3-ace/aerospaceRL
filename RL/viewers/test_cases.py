'''
Runs test cases

Created by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

### TEST CASE ###
Case = 'NoRTA'
# Case = 'Velocity'
# Case = 'Simplex'
# Case = 'IASIF'
#################
### TRAINING ####
Train = 'NoRTA'
# Train = 'Velocity'
# Train = 'Simplex'
# Train = 'IASIF'
#################
##### RANGE #####
RANGE = 1000 # m
# RANGE = 10000 # m
#################

import gym
import dubins_gym
import os
import core_spinup as core
import math
import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16}) # For Presentation/Paper
plt.rcParams.update({'figure.autolayout': True})

# Assumes spacecraftdockingrl is in your home directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'PPO-spinup/asif'))
PATH = os.path.expanduser("~") + "/spacecraftdockingrl/RL_algorithms/saved_models"
if not os.path.isdir(PATH):
    print('PATH ISSUE - UPDATE YOUR PATH')
    exit()

#### MODEL PATHS ####
NoRTA_model = f"{PATH}/NoRTA1.dat"
Velocity_model = f"{PATH}/Velocity1.dat"
Simplex_model = f"{PATH}/ISimplex1.dat"
IASIF_model = f"{PATH}/IASIF1.dat"
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
if Case == 'Velocity':
    from Simple_velocity_limit import ASIF
elif Case == 'Simplex':
    from ISimplex import ASIF
elif Case == 'IASIF':
    from IASIF import ASIF

if Case == 'Velocity' or Case == 'Simplex' or Case == 'IASIF':
    # Call ASIF class
    asif = ASIF(env)

    # Define action
    def asif_act(obs, act):
        # Clip action to be within accepted range
        act = np.clip(act, -env.force_magnitude, env.force_magnitude)
        # Rearrange observation state vector
        x0 = [obs[0], obs[1], 0, obs[2], obs[3], 0]
        # Rearrange action vector
        u_des = np.array([[act[0]], [act[1]], [0]])
        # Call asif function
        u = asif.main(x0, u_des)
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
ac_kwargs = dict(hidden_sizes=[64,64])
ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
# Load appropriate model
if Train == 'NoRTA':
    ac.load_state_dict(torch.load(NoRTA_model))
elif Train == 'Velocity':
    ac.load_state_dict(torch.load(Velocity_model))
elif Train == 'Simplex':
    ac.load_state_dict(torch.load(Simplex_model))
elif Train == 'IASIF':
    ac.load_state_dict(torch.load(IASIF_model))

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
        if Case == 'Velocity' or Case == 'IASIF' or Case == 'Simplex':
            act = asif_act(obs,act)
        # Take step in environment
        obs, _, done, _ = env.step(act)
        # Track if velocity violated constraint (No RTA)
        if Case == 'NoRTA':
            over_max_vel, _, _ = env.check_velocity(act[0], act[1])
            if over_max_vel:
                RTA_percent += 1
        # Track if RTA is on
        elif Case == 'Velocity' or Case == 'IASIF' or Case == 'Simplex':
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
