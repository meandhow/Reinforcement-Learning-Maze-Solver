# Reinforcement-Learning-Maze-Solver

To run the program, execute train_and_test.py

This repository contains code generating a random maze environment and training Reinofrcement Learning agent to reach the goal in the maze. coursework_specification contains description of all files. The description below is regarding agent.py file, which contains the learning and behaviours of the agent

The implemented solution for solving the maze environment is Deep Q-Learning with Replay Buffer and Target Network. It additionally contains an evaluation check after each training episode that can stop the training given the agent reaches the goal with greedy policy. The hyper parameters were hand-tuned using trial and error method combined with understanding of the underlying principles. Alternative implementation included a variation of Prioritised Experience Replay, however it was found that it actually worsen the performance of the network due to added computational complexity. The implementation is outlined below.

Lines 8-24 contain Network class. This class allows for initiation of the neural net- work with given architecture using torch module. Additionally, the forward function is defined which is responsible for gaining network’s output given an input.

Lines 27-68 contain DQN class. This class is responsible for training of the neural net. The two instances of the Network class are initiated with respective optimisers, the regular q network and the target network. Further functions are responsible for calculation of the loss and its back-propagation. The mean-square-error loss of the mini-batch is calculated using Bellman equation, with incorporated target network for successive state-action value prediction.

Lines 70-116 contain ReplayBuffer class. This class allows for creation of container where the agent’s transition will be put into. It contains methods for appending the transition to the buffer (container) and for sampling of random mini-batch for training purposes.

Lines 123-260 contain the main body of code in form of Agent class. Firstly, the instances of DQN and ReplayBuffer are initiated, along with multiple numerical and flag variables. Function ”has finished episode” is responsible for checking whether the episode is done and additionally it initiates the greedy run evaluating the policy every other episode. Function ”get greedy action” calculates the greedy action by taking the argmax on forward pass through neural. Function ”get epsilon function” can either return the greedy function, with the same mechanism as the previously mentioned function, or a randomized action. The randomized action has respectively 35, 30, 30 and 5 percent chance of being right, up, down and left. This distribution favorizes actions leading to the goal state. Epsilon is decaying with the number of episodes, in order to ensure high exploration at the start of training and then higher specialization in getting the right values for optimal path states. The function ”get next action” checks whether the agent is in training or evaluation mode and then the respective epsilon-greedy or greedy action is gotten. The function ”set next state and distance” is responsible for training of the agent. Firstly, it calculates the reward based on the supplied distance to goal. The reward function was tuned to give better value differentiation near the initial state. Next, if the agent is in training mode, the transition is appended to the buffer and a mini-batch is randomly sampled from it. Based on that batch, the prediction loss is calculated and back propagated through network for learning. Finally, the target network is updated with set frequency.

Created as submission for Reinforcement Learning Coursework 2 at Imperial College London
