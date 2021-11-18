from setuptools import setup

setup(name='aero_gym',
	version='0.1',
	install_requires=[
		'gym',
		'mpi4py',
		'tensorflow>=1.8.0,<2.0',
		'torch',
		'joblib',
		'matplotlib',
		'psutil',
		'tqdm',
		'seaborn==0.8.1',
		'quadprog==0.1.8'
	]
)
