[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic. The goal was to get an average score of +13 over 100 consecutive episodes.

### Features
- N-step DQN (Sutton, 1988)
- Double DQN (van Hasselt, Guez, and Silver, 2015)
- Dueling DQN (Wang et al., 2015)
- Noisy Linear Layers (Fortunanto et al., 2017)

### Deep Neural Network - Architecture
Input(4) ->
Linear Layer(128) ->
Linear Layer(64)

**Stream 1:**
-> Noisy Linear Layer(64)
-> Noisy Linear Layer(4)

**Stream 2:**
-> Noisy Linear Layer(64)
-> Noisy Linear Layer(1)

### Getting Started

#### Getting the Unity-ML environment
1. The environment can be downloade from the links below:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
2. Place the file in the root directory of the repository and unzip the file.
 
#### Requirements
1. Install anaconda [click here](https://conda.io/docs/user-guide/install/index.html)
2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drl python=3.6
	source activate drl
	```
	- __Windows__: 
	```bash
	conda create --name drl python=3.6 
	activate drl
	```

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

4. Clone the repository (if you haven't already!). Then, install several dependencies.
```bash
git clone https://github.com/meiermark/rl-navigation.git
cd rl-navigation/ml-agents
pip install .
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl` environment.  
```bash
python -m ipykernel install --user --name drl --display-name "drl"
```

6. Before running code in the notebook, change the kernel to match the `drl` environment by using the drop-down `Kernel` menu. 


### Start the training
The training can be done in `Navigation.ipynb`  


### TODO - Outlook
- Prioritized replay (segmentation tree already done)
- Categorical DQN