
[image1]: https://raw.githubusercontent.com/meiermark13/rl-navigation/master/training_results.png "Training"



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

### Parameters
- Linear noisy layers are initialized with a sigma of 0.017
- replay memory of size 1e5
- training on mini-batches with size 64
- tau of 1e-3 for updating the target network
- a learning rate of 5e-4 for the optimizer 
    - The learning rate is small enough to be used for all episodes
- training/updating the network every 4 steps
- step size of 2 for the cumulative discounted reward with the n-step buffer 
- gamma (=discount factor) of 0.99 for the cumulative discounted reward
- no epsilon due to the noisy linear layers

### Training performance

![Trained Agent][image1]

The DQN achieves an average reward (over 100) episodes of 13 after 300 episodes.
Between 500 and 600 episodes, a maximum average reward of 16.62 is measured.
The corresponding model parameters are saved in `checkpoint.pth`. 