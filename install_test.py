import gym
import aero_gym

# env = gym.make('dubins-aircraft-v0')
# env = gym.make('dubins-aircraft-continuous-v0')
# env = gym.make('spacecraft-docking-v0')
env = gym.make('spacecraft-docking-continuous-v0')

done = False
env.reset()
steps = 0
while not done:
	steps += 1
	obs, rew, done, info = env.step(env.action_space.sample())
	if steps % 4 == 0:
		env.render()

env.close()
print('Properly Installed!')
