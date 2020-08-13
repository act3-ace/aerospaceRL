# Custom Environments
Created by Kyle Dunlap and Kai Delsing  
Mentor: Kerianne Hobbs  

## Installation
1. Install Anaconda (Recommended):  
Follow the installation instructions for Anaconda 3 [here](https://docs.continuum.io/anaconda/install/).  
2. Create a conda Python 3.6 environment, which will help organize the packages used:  
`conda create -n aero python=3.6`  
3. To use Python in this environment, activate it by running:  
`conda activate aero`  
4. Install OpenMPI (Ubuntu/Debian):  
`sudo apt-get update && sudo apt-get install libopenmpi-dev`  
5. In the directory you want to save the environment, run the command:  
`git clone [copy link here]`
6. Then run the commands:  
`cd spacecraftpublicrelease`  
`pip install -e .`  

To test your installation, run the `install_test.py` file. You should see a rendering of the spacecraft docking environment.  

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
3. Navigate to the 'spacecraftpublicrelease/RL_algorithms' folder.
4. Decide what algorithm you'd like to run, e.g. run the PPO RL algorithm `python PPO.py`
