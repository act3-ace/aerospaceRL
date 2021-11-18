'''
Plots TensorBoard data using matplotlib
Used with Spacecraft Docking Environment

Created by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_runs')

plt.rcParams.update({"text.usetex": True,'font.size': 20, 'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']})
plt.rcParams.update({'figure.autolayout': True})

def main(FILE=['NoRTA1'], plots=['Delta-V'], cutoff=2000, smooth=11):

	"""
	FILE: TB files to use
	plots: TB plots to display (Use name from TensorBoard)
	cutiff: Number of epochs to display
	smooth: Number of episodes to smooth over
	"""

	# Used to move to next figure
	fig = 0

	# Cycles through tensorboard tags
	for tag in plots:
		fig += 1
		# Define ylabel and title
		if tag == 'Delta-V':
			ylabel = '$\Delta V$ [m/s]'
			title = 'Average $\Delta V$'
		elif tag == 'Episode-Length':
			ylabel = 'Time [s]'
			title = 'Average Episode Length'
		elif tag == 'RTA-on-percent':
			ylabel = 'Percent [\%]'
			title = 'Average Percent of Time Constraint is Violated'
		elif tag == 'Return':
			ylabel = 'Total Reward'
			title = 'Average Episode Return'
		elif tag == 'Success-Rate':
			ylabel = 'Percent [\%]'
			title = 'Average Successful Episodes'
		else:
			print('Invalid Plot Name')

		# Set variables
		x2 = []
		x3 = []
		x4 = []
		x5 = []
		x6 = []
		x7 = []

		# Cycles through TB runs
		for file in FILE:
			# Initialize x
			x = []
			# For TB file
			for event in tf.compat.v1.train.summary_iterator(f"{PATH}/{file}"):
				# For each epoch
				for value in event.summary.value:
					# If tag corresponds to desired plot, append value to x
					if value.tag == tag:
						x.append(value.simple_value)
					# Break after specified number of epochs
					if len(x) > cutoff+1:
						break
				if len(x) > cutoff+1:
					break

			# Cutoff x at desired epoch
			x = x[:cutoff+1]

			# Smooth x
			x = np.convolve(x,np.ones(smooth),'same')/ np.convolve(np.ones(len(x)),np.ones(smooth),'same')

			# Append to proper training file
			if file == 'NoRTAHP1' or file == 'NoRTAHP2' or file == 'NoRTAHP3' or file == 'NoRTAHP4' or file == 'NoRTAHP5':
				x2.append(x)
			elif file == 'NoRTA1' or file == 'NoRTA2' or file == 'NoRTA3' or file == 'NoRTA4' or file == 'NoRTA5':
				x3.append(x)
			elif file == 'ExS1' or file == 'ExS2' or file == 'ExS3' or file == 'ExS4' or file == 'ExS5':
				x4.append(x)
			elif file == 'ImS1' or file == 'ImS2' or file == 'ImS3' or file == 'ImS4' or file == 'ImS5':
				x5.append(x)
			elif file == 'ExO1' or file == 'ExO2' or file == 'ExO3' or file == 'ExO4' or file == 'ExO5':
				x6.append(x)
			elif file == 'ImO1' or file == 'ImO2' or file == 'ImO3' or file == 'ImO4' or file == 'ImO5':
				x7.append(x)


		# Plot results
		plt.figure(fig)
		if len(x2) != 0:
			sns.tsplot(data=x2, color='r')
		if len(x3) != 0:
			sns.tsplot(data=x3, color='tab:orange')
		if len(x4) != 0:
			sns.tsplot(data=x4, color='y')
		if len(x5) != 0:
			sns.tsplot(data=x5, color='tab:green')
		if len(x6) != 0:
			sns.tsplot(data=x6, color='b')
		if len(x7) != 0:
			sns.tsplot(data=x7, color='tab:purple')
		plt.xlabel('Environment Interactions (10e6)')
		plt.xticks([0, 353, 706, 1059, 1412, 1765, 2117],[0, 30, 60, 90, 120, 150, 180])
		plt.ylabel(ylabel)
		plt.grid(True)
		# plt.title(title)

	# Plot legend separately
	plt.figure(fig+1)
	plt.plot(0,0,color='r', linewidth=2)
	plt.plot(0,0,color='tab:orange', linewidth=2)
	plt.plot(0,0,color='y', linewidth=2)
	plt.plot(0,0,color='tab:green', linewidth=2)
	plt.plot(0,0,color='b', linewidth=2)
	plt.plot(0,0,color='tab:purple', linewidth=2)
	plt.axis('off')
	plt.legend(['No RTA - HP','No RTA','Explicit Switching','Implicit Switching','Explicit Optimization','Implicit Optimization'], loc='upper center')

	plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--Files', nargs='+', default=['NoRTAHP1','NoRTAHP2','NoRTAHP3','NoRTAHP4','NoRTAHP5','NoRTA1','NoRTA2','NoRTA3','NoRTA4','NoRTA5','ExS1','ExS2','ExS3','ExS4','ExS5','ImS1','ImS2','ImS3','ImS4','ImS5','ExO1','ExO2','ExO3','ExO4','ExO5','ImO1','ImO2','ImO3','ImO4','ImO5']) # TB files to use
	parser.add_argument('--Plots', nargs='+', default=['Delta-V','Episode-Length','RTA-on-percent','Return','Success-Rate']) # TB plots to display
	parser.add_argument('--cutoff', type=int, default=2000) # Number of epochs to display
	parser.add_argument('--smooth', type=int, default=11) # Number of episodes to smooth over
	args = parser.parse_args()

	main(FILE=args.Files, plots=args.Plots, cutoff=args.cutoff, smooth=args.smooth)

'''
Example of how to run TensorBoard_Plot in terminal from home directory, plot only delta-v with 50 episodes smoothing:
python aerospacerl/RL/viewers/TensorBoard_Plot.py --Plots Delta-V --smooth 50
'''
