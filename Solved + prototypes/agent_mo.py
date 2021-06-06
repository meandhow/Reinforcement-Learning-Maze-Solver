# Import some modules from other libraries
import numpy as np
import torch
import time
import random
from matplotlib import pyplot as plt
import cv2
plt.ion()
# Import the environment module
from environment import Environment
import collections
import statistics

class QValueVisualiser:

    def __init__(self, environment, magnification=500):
        self.environment = environment
        self.magnification = magnification
        self.half_cell_length = 0.05 * self.magnification
        # Create the initial q values image
        self.q_values_image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    def draw_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)
        # Loop over the grid cells and actions, and draw each q value
        for col in range(10):
            for row in range(10):
                # Find the q value ranges for this state
                max_q_value = np.max(q_values[col, row])
                min_q_value = np.min(q_values[col, row])
                q_value_range = max_q_value - min_q_value
                # Draw the q values for this state
                for action in range(4):
                    # Normalise the q value with respect to the minimum and maximum q values
                    q_value_norm = (q_values[col, row, action] - min_q_value) / q_value_range
                    # Draw this q value
                    x = (col / 10.0) + 0.05
                    y = (row / 10.0) + 0.05
                    self._draw_q_value(x, y, action, float(q_value_norm))
        # Draw the grid cells
        self._draw_grid_cells()
        # Show the image
        cv2.imwrite('q_values_mine.png', self.q_values_image)
        cv2.imshow("Q Values", self.q_values_image)
        cv2.waitKey(0)

    def _draw_q_value(self, x, y, action, q_value_norm):
        # First, convert state space to image space for the "up-down" axis, because the world space origin is the bottom left, whereas the image space origin is the top left
        y = 1 - y
        # Compute the image coordinates of the centre of the triangle for this action
        centre_x = x * self.magnification
        centre_y = y * self.magnification
        # Compute the colour for this q value
        colour_r = int((1 - q_value_norm) * 255)
        colour_g = int(q_value_norm * 255)
        colour_b = 0
        colour = (colour_b, colour_g, colour_r)
        # Depending on the particular action, the triangle representing the action will be drawn in a different position on the image
        if action == 0:  # Move right
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y - self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 1:  # Move up
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = centre_x - self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 2:  # Move left
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y + self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 3:  # Move down
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = centre_x + self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    def _draw_grid_cells(self):
        # Draw the state cell borders
        for col in range(11):
            point_1 = (int((col / 10.0) * self.magnification), 0)
            point_2 = (int((col / 10.0) * self.magnification), int(self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        for row in range(11):
            point_1 = (0, int((row / 10.0) * self.magnification))
            point_2 = (int(self.magnification), int((row / 10.0) * self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition
    def optimal_step(self):
        # Choose the next action.
        discrete_action = self._choose_optimal_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def epsilon_step(self, iterations):
        # Choose the next action.
        discrete_action, random_prob = self._choose_epsilon_action(iterations)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition, random_prob

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return random discrete action from 0 to 3
        action = random.randint(0,3)
        return action

    def _choose_optimal_action(self):
        # optimal action
        state_tensor=torch.tensor(self.state).unsqueeze(0)
        action = torch.argmax(dqn.q_network.forward(state_tensor)).item()
        return action
    def _choose_epsilon_action(self, iteration):
        # epsilon greedy function
        # epislon=1/((iteration+1)**0.2)
        # eq = 1/(-2+((iteration+0.1)*0.007)**0.4)
        # eq = 1/((iteration+1)**0.6)
        eq = 1/((iteration+1)*0.4)

        if eq < 0:
            epsilon = 1
        else:
            epsilon=eq
        if random.uniform(0,1) > epsilon:
            state_tensor=torch.tensor(self.state).unsqueeze(0)
            action = torch.argmax(dqn.q_network.forward(state_tensor)).item()
        else:
            action = random.randint(0,3)
        return action, epsilon


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        #NESW 0,1,2,3

        if discrete_action == 0:
            # right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            # up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2:
            # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # Move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        
        # if discrete_action == 0:
        #     # right
        #     continuous_action = np.array([0, 0.1], dtype=np.float32)
        # elif discrete_action == 1:
        #     # up
        #     continuous_action = np.array([0.1, 0], dtype=np.float32)
        # elif discrete_action == 2:
        #     # Move left
        #     continuous_action = np.array([0, -0.1], dtype=np.float32)
        # elif discrete_action == 3:
        #     # Move down
        #     continuous_action = np.array([-0.1, 0], dtype=np.float32)
        
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network= Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.005)
        self.optimiser_target = torch.optim.Adam(self.target_network.parameters(), lr=0.005)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network_transition(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_online(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def train_q_network_minibatch(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_minibatch(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def train_q_network_bellman_minibatch(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_bellman_minibatch(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def train_q_network_bellman_minibatch_loss_with_target(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_bellman_minibatch_with_target(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _calculate_loss_online(self, transition):
        #transition = (self.state, discrete_action, reward, next_state)
        state_tensor=torch.tensor(transition[0], dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor(transition[2], dtype=torch.float32)
        predicted_q_value_tensor = self.q_network.forward(state_tensor)[0,transition[1]]
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, reward_tensor)
        return loss

    def _calculate_loss_minibatch(self, minibatch):
        #transition = (self.state, discrete_action, reward, next_state)
        #unzip the values in the minibatch
        states, actions, rewards, *others = zip(*minibatch)

        state_tensor=torch.tensor(states, dtype=torch.float32)
        action_tensor=torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        predicted_q_value_tensor = self.q_network.forward(state_tensor).gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, reward_tensor)
        return loss

    def _calculate_loss_bellman_minibatch(self, minibatch, gamma=0.9):
        #transition = (self.state, discrete_action, reward, next_state)
        #unzip the values in the minibatch
        states, actions, rewards, next_states = zip(*minibatch)

        state_tensor=torch.tensor(states, dtype=torch.float32)
        next_state_tensor=torch.tensor(next_states, dtype=torch.float32)
        action_tensor=torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        predicted_q_value_tensor = self.q_network.forward(state_tensor).gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        succesor_q_value_tensor = torch.amax(self.q_network.forward(next_state_tensor),1)
        label = reward_tensor + gamma * succesor_q_value_tensor 
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, label)
        return loss
    
    def _calculate_loss_bellman_minibatch_with_target(self, minibatch, gamma=0.9):
        #transition = (self.state, discrete_action, reward, next_state)
        #unzip the values in the minibatch
        states, actions, rewards, next_states = zip(*minibatch)

        state_tensor=torch.tensor(states, dtype=torch.float32)
        next_state_tensor=torch.tensor(next_states, dtype=torch.float32)
        action_tensor=torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        predicted_q_value_tensor = self.q_network.forward(state_tensor).gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        succesor_q_value_tensor = torch.amax(self.target_network.forward(next_state_tensor),1)
        succesor_q_value_tensor = succesor_q_value_tensor.detach()
        label = reward_tensor + gamma * succesor_q_value_tensor 
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, label)
        return loss

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch):
        total_loss=[]
        for transition in minibatch:
            s=transition[0]
            # s.append(transition[0])
            a=transition[1]
            s_tensor = torch.tensor(s)
            reward = transition[2]
            s_prime=transition[3]
            s_prime_tensor = torch.tensor(s_prime)
            print(self.q_network.forward(s_tensor))
            loss = pow((reward - self.q_network.forward(s_tensor)[a].item()),2) #+ gamma*torch.max(self.q_network.forward(s_prime_tensor)).item()
            total_loss.append(loss)
        MSE_loss=statistics.mean(total_loss)
        print(MSE_loss)
        loss = torch.tensor(MSE_loss,requires_grad=True)
        # print("loss:")
        # print(loss)
        # print("end_loss")
        return loss

    def populate_q_array(self,q_values):
        for col in range(10):
            for row in range(10):
                x_coordinate = row/10.0+0.05
                y_coordinate = row/10.0+0.05
                xy=[x_coordinate,y_coordinate]
                xy_tensor=torch.tensor(xy, dtype=torch.float32).unsqueeze(0)
                xy_value = self.q_network.forward(xy_tensor)
                for action in range(4):
                    q_values[col,row,action]=xy_value[0,action].item() #[action].item()
        return q_values

    def update_target_network(self):
        q_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(q_dict)
        

class ReplayBuffer:
    def __init__(self, maxlength):
        # self.length = length
        self.replay_buffer = collections.deque(maxlen=maxlength)
        # self.batch_size = size
    def buffer_size(self):
        return len(self.replay_buffer)
    def append_transition(self, transition):
        self.replay_buffer.append(transition)

    def rand_mini_batch(self, size):
        # minibatch_indices = np.random.choice(range(len(self.replay_buffer)), size)
        # print(minibatch_indices)
        minibatch=[]
        if len(self.replay_buffer) > size:
            minibatch_indices = np.random.choice(range(len(self.replay_buffer)), size)
            for i in range(size):
                minibatch.append([])
        else:
            minibatch_indices = np.random.choice(range(len(self.replay_buffer)), len(self.replay_buffer))
            for i in range(len(self.replay_buffer)):
                minibatch.append([])
        # print(minibatch_indices)
        for i in range(len(minibatch)):
            index = minibatch_indices[i]
            minibatch[i]=np.array(self.replay_buffer[index])

        return minibatch

# def optimal_policy_trace(length):
#     trace=[]
#     state=[]
#     initial_state=environment.init_state
#     state.extend(initial_state)
#     for step in range(length):
#         trace.append([])
#         trace[step]=state
#         state_tensor=torch.tensor(state)
#         action = torch.argmax(dqn.q_network.forward(state_tensor)).item()
#         action_step = agent._discrete_action_to_continuous(action)
#         if state[0]+ action_step[0] < 0 or state[1]+ action_step[1] < 0:
#             new_state=state
#         else:
#             new_state=[state[0]+ action_step[0],state[1]+ action_step[1]]
#         state=new_state
#     return trace
def optimal_policy_trace(length):
    agent.reset()
    trace=[]
    for step in range(length):
        transition = agent.optimal_step()
        trace.append([transition[0],transition[3]])
        # if round(transition[3][0],2) == environment.goal_state[0] and round(transition[3][1],2) == environment.goal_state[1]:
        #     break
    return trace
def random_policy_trace(length):
    agent.reset()
    trace=[]
    for step in range(length):
        transition = agent.step()
        trace.append([transition[0],transition[3]])
    return trace
# Main entry point
if __name__ == "__main__":
    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    dqn.update_target_network()

    # Create a visualiser
    visualiser = QValueVisualiser(environment=environment, magnification=500)

    buffer = ReplayBuffer(maxlength=50000)
    losses = []
    iterations = []
    random_probs = []
    goal_reached_list = []
    # Create a graph which will show the loss as a function of the number of training iterations
    fig, (ax_loss,ax_prob, ax_goal) = plt.subplots(3,1, sharex=True)
    plt.subplots_adjust(hspace = 0.35)

    ax_loss.set(xlabel='Episode', ylabel='Loss', title='1.2(b) Loss Curve for Mini-Batch + Bellman + Target Network')
    ax_prob.set(xlabel='Episode', ylabel='Probability', title='Probability of chosen action being random')
    ax_goal.set(xlabel='Episode', ylabel='True(1) or False (0)', title='Goal reached?')

    # q_values = dqn.populate_q_array(np.zeros([10, 10, 4]))
    # print(q_values)
    # # Draw the image
    # visualiser.draw_q_values(q_values)
    # trace=optimal_policy_trace(100)
    # print(trace)
    # environment.draw_optimal_policy(trace,steps=100)

    # Loop over episodes
    for episode in range (50):
        # Reset the environment for the start of the episode.
        agent.reset()
        episode_losses=[]
        goal_reached = 0
        if episode%2 == 0:
            dqn.update_target_network()
        # Loop over steps within this episode. The episode length here is 20.
        # for step_num in range(int(1000 * 1/(episode+1))):
        for step_num in range(500):
            # Step the agent once, and get the transition tuple for this step
            transition, random_prob = agent.epsilon_step(episode)
            current_state = transition[3]
            goal_state = environment.goal_state
            if goal_reached == 1:
                continue
            else:
                if goal_state[0] - 0.15 < current_state[0] < goal_state[0] + 0.15 and goal_state[1] - 0.15 < current_state[1] < goal_state[1] + 0.15:
                    goal_reached = 1
                else:
                    goal_reached = 0
            # print(transition)
            buffer.append_transition(transition)

            if buffer.buffer_size() >= 200:
                minibatch = buffer.rand_mini_batch(200)
                # loss_value = dqn.train_q_network_bellman_minibatch(minibatch)
                loss_value = dqn.train_q_network_bellman_minibatch_loss_with_target(minibatch)
                episode_losses.append(loss_value)
        # Update the list of iterations
        # Plot and save the loss vs iterations graph
        if len(episode_losses) != 0:
            iterations.append(episode)
            losses.append(statistics.mean(episode_losses))
            transition, random_prob = agent.epsilon_step(episode)
            random_probs.append(random_prob)
            goal_reached_list.append(goal_reached)
            print('Episode ' + str(episode) + ', Loss = ' + str(losses[-1]) + ', Rand_prob = ' + str(random_prob) + ', Goal_reached = ' + str(goal_reached))
            ax_loss.plot(iterations, losses, color='blue')
            ax_loss.set_yscale('log')
            ax_prob.plot(iterations, random_probs, color='red')
            # ax_prob.set_yscale('log')
            ax_goal.plot(iterations, goal_reached_list, color='green', )
            ax_goal.set_ylim(-0.25,1.25)

            # plt.yscale('log')
        # plt.show()


    q_values = dqn.populate_q_array(np.zeros([10, 10, 4]))
    visualiser.draw_q_values(q_values)

    trace=optimal_policy_trace(20)
    print(trace)
    environment.draw_optimal_policy(trace)
