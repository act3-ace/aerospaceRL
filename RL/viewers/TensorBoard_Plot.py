'''
Plots TensorBoard data using matplotlib

Created by: Kyle Dunlap
Mentor: Kerianne Hobbs
'''

#### MODELS ####
FILE = [
'NoRTA1',
'NoRTA2',
'Velocity1',
'Velocity2',
'IASIF1',
'IASIF2',
'ISimplex1',
'ISimplex2'
]
################
### TB PLOTS ###
plots = [
'Delta-V',
'Episode-Length',
'RTA-on-percent',
'Return',
'Success-Rate'
]
################
#### EPOCHS ####
cutoff = 2000
################
### SMOOTHING ##
smooth = 11
################


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# plt.rcParams.update({'font.size': 16}) # For Presentation
plt.rcParams.update({'font.size': 20}) # For Paper
plt.rcParams.update({'figure.autolayout': True})

# Assumes aerospacerl is in your home directory
PATH = os.path.expanduser("~") + "/aerospacerl/RL/saved_runs"
if not os.path.isdir(PATH):
	print('PATH ISSUE - UPDATE YOUR PATH')
	exit()

# Used to move to next figure
fig = 0

# Cycles through tensorboard tags
for tag in plots:
	fig += 1
	# Define ylabel and title
	if tag == 'Delta-V':
		ylabel = r'$\Delta$ V (m/s)'
		title = r'Average $\Delta$ V'
	elif tag == 'Episode-Length':
		ylabel = 'Time (sec)'
		title = 'Average Episode Length'
	elif tag == 'RTA-on-percent':
		ylabel = 'Percent (%)'
		title = 'Average Percent of Time Constraint is Violated'
	elif tag == 'Return':
		ylabel = 'Total Reward'
		title = 'Average Episode Return'
	elif tag == 'Success-Rate':
		ylabel = 'Percent (%)'
		title = 'Average Successful Episodes'
	else:
		print('Invalid Plot Name')

	# Set variables
	x2 = []
	x3 = []
	x4 = []
	x5 = []

	# Cycles through TB runs
	for file in FILE:
		# Initialize x
		x = []
		# For TB file
		for event in tf.train.summary_iterator(f"{PATH}/{file}"):
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
		if file == 'NoRTA1' or file == 'NoRTA2':
			x2.append(x)
		elif file == 'Velocity1' or file == 'Velocity2':
			x3.append(x)
		elif file == 'ISimplex1' or file == 'ISimplex2':
			x4.append(x)
		elif file == 'IASIF1' or file == 'IASIF2':
			x5.append(x)


	# Plot results
	plt.figure(fig)
	if len(x2) != 0:
		sns.tsplot(data=x2, color='r')
	if len(x3) != 0:
		sns.tsplot(data=x3, color='b')
	if len(x4) != 0:
		sns.tsplot(data=x4, color='y')
	if len(x5) != 0:
		sns.tsplot(data=x5, color='m')
	plt.xlabel('Epoch')
	plt.ylabel(ylabel)
	plt.grid(True)
	# plt.title(title)

# Plot legend separately
plt.figure(fig+1)
plt.plot(0,0,color='r', linewidth=2)
plt.plot(0,0,color='b', linewidth=2)
plt.plot(0,0,color='y', linewidth=2)
plt.plot(0,0,color='m', linewidth=2)
plt.axis('off')
plt.legend(['Training with No RTA','Training with Velocity Limit','Training with Simplex','Training with ASIF'], loc='upper center')

plt.show()