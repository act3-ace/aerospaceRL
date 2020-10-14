# Custom Aerospace Reinforcement Learning Environments
Created by Kyle Dunlap, Kai Delsing, and Mark Mote
Mentor: Kerianne Hobbs 

Approved for public release: distribution unlimited. Case Number 88ABW-2020-2644.


## Installation
1. Install Anaconda (Recommended):  
Follow the installation instructions for Anaconda 3 [here](https://docs.continuum.io/anaconda/install/).  
2. Create a conda Python 3.7 environment, which will help organize the packages used:  
`conda create -n aero python=3.7`  
3. To use Python in this environment, activate it by running:  
`conda activate aero`  
4. Install OpenMPI (Ubuntu/Debian):  
`sudo apt-get update && sudo apt-get install libopenmpi-dev`  
5. Install Python header files:  
`sudo apt-get install python-dev`  
6. In the directory you want to save the environment, copy the cloning link and run the command:  
`git clone [copy link here]`
7. Then run the commands:  
`cd aerospacerl`  
`pip install -e .`  
If you get a memory error while running the previous command, run the following command:  
`pip install -e . --no-cache-dir`

To test your installation, run the `install_test.py` file. You should see a rendering of the spacecraft docking environment.  

## Gurobi Installation (Required for some RTA algorithms)  
In order to use the IASIF and CBF_for_speed_limit RTA algorithms, install Gurobi Optimization using the following steps. A license is required. Academic licenses are free.  
1. In your terminal, make sure the aero conda environment is activated. Activate it by running:  
`conda activate aero`  
2. Add Gurobi to your default search list:  
`conda config --add channels http://conda.anaconda.org/gurobi`  
3. Install Gurobi:  
`conda install gurobi`  
4. Obtain a license [here](https://www.gurobi.com/documentation/9.0/quickstart_linux/retrieving_and_setting_up_.html#section:RetrieveLicense).  

## Using the custom environments (In Python)
There are two custom environments: `dubins_aircraft.py` and `spacecraft_docking.py`, each with a discrete and continuous action space.  The code for these environments is located at `dubins_gym/envs/`  

To use the custom environments, in __Python__ first import the gym and aero_gym libraries:  
```python
import aero_gym
```  
Then use the `gym.make()` function to make the environment one of the following:  
```python
# Discrete Action Space
gym.make('dubins-aircraft-v0')
gym.make('spacecraft-docking-v0')
# Continuous Action Space
gym.make('dubins-aircraft-continuous-v0')
gym.make('spacecraft-docking-continuous-v0')
```

## Running from the Command Line
These files assume that you have Anaconda or equivalent and standard Python packages installed.  
1. To run from the command line, open a terminal  
2. To run using the correct Anaconda packages, type `conda activate aero`. This should make (aero) show up before you username.  
3. For example, run PPO using the spacecraft-docking-continuous-v0 environment for 10 epochs:  
`python aerospacerl/RL/PPO.py --epochs 10`
