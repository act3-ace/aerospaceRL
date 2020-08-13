import gym
import aero_gym

# env = gym.make('dubins-aircraft-v0')
# env = gym.make('dubins-aircraft-continuous-v0')
# env = gym.make('spacecraft-docking-v0')
env = gym.make('spacecraft-docking-continuous-v0')

done = False
env.reset()
while not done:
	obs, rew, done, info = env.step(env.action_space.sample())
	env.render()

env.close()
print('Properly Installed!')
