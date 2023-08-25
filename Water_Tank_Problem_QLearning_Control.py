import math
import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt


##########################
# Dynamics of Water Tank #
##########################

class WaterTank:
    def __init__(self):
        self.radius = 0.5
        self.area = math.pi * self.radius ** 2
        self.k = 1/10

    def dhdt(self, h, t, v_in, v_out_uncertainty):
        ''' 
        Differential equation describing the water level change 
        '''
        v_out = h * self.k + v_out_uncertainty
        return 1/self.area * (v_in - v_out)

    def ode_solver(self, h0, v_in, t):
        ''' 
        Solver for the differential equation
        '''
        t = np.linspace(t, t+1, 2)
        v_out_uncertainty = np.random.normal(0, 0.5, 1)  # random outflow
        h_array = odeint(self.dhdt, h0, t, args=(v_in, v_out_uncertainty[0]))
        h = np.round(h_array[-1][0], 1)
        if h < 7:
            h = 7
        elif h > 13:
            h = 13
        return h

    def episode_end(self, t):
        ''' 
        Allows the agent to control the water level for 10 time-steps 
        After that the episode is over
        '''
        if t >= 10:
            return True

    def reward_gen(self, action, next_state):
        ''' 
        Reward function 
        '''
        if action > 1.5:  # penalize larger change in control action
            reward = - (10 - next_state) ** 2 - 6 * action ** 2
        else:
            reward = - (10 - next_state) ** 2

        return reward


####################
# Q-learning agent #
####################

class QLearningAgent(WaterTank):
    def __init__(self, water_tank, all_states, all_actions, all_states_dict, q_table):
        super().__init__()
        self.water_tank = water_tank
        self.all_states = all_states
        self.all_actions = all_actions
        self.all_states_dict = all_states_dict
        self.q_table = q_table

    def get_next_state(self, state, action, t):
        ''' 
        Get the next state based on current state and action
        Use of ODE solver to generate next state 
        '''
        max_state = np.max(self.all_states)
        min_state = np.min(self.all_states)
        next_state = self.water_tank.ode_solver(state, action, t)

        # bounds for valid states
        if np.isnan(next_state):
            next_state = state
        if next_state > max_state:
            next_state = max_state
        elif next_state < min_state:
            next_state = min_state

        return next_state

    def next_action(self, state, epsilon):
        ''' 
        Select the next action based on epsilon-greedy policy 
        '''
        if np.random.random() < epsilon:
            action = random.choice(self.all_actions.tolist())
        else:
            state_index = self.all_states_dict[state]
            if np.all(self.q_table[state_index] == initial_q_values):
                action = random.choice(self.all_actions.tolist())
            else:
                action = np.argmax(self.q_table[state_index])
                action = self.all_actions.tolist()[action]
        return action

    def key_and_value(self, input_dict, target_key):
        ''' 
        Find key and value in a dictionary 
        Allows for index of state action pair 
        '''
        for key, value in input_dict.items():
            if key == target_key:
                return key, value

        return None, None

    #### Training the agent ####
    def q_train(self, num_episodes, h0, v0, EPSILON, GAMMA, ALPHA, terminal_state):
        for episode in range(num_episodes):
            t = 0
            height_initial = self.water_tank.ode_solver(h0, v0, t)
            state, state_number = self.key_and_value(self.all_states_dict, height_initial)

            while not self.episode_end(t):
                t += 1
                action = self.next_action(state, EPSILON)
                action, action_number = self.key_and_value(all_actions_dict, action)

                new_state_array = self.get_next_state(state, action, t)
                new_state = np.round(new_state_array, 1)
                new_state_index = self.all_states_dict[new_state]

                reward = self.reward_gen(action, new_state)
                old_q_value = self.q_table[state_number, action_number]
                temporal_difference = reward + (GAMMA * np.max(self.q_table[new_state_index])) - old_q_value

                new_q_value = old_q_value + (ALPHA * temporal_difference)
                self.q_table[state_number, action_number] = new_q_value
                state = new_state
                state_number = new_state_index
                print(f"\r {episode / num_episodes * 100}%", end="")

        print('\n')
        print('---------------- Q-Table ----------------')
        print(q_table)

    #### Plotting functions ####
    def plot_state_vs_best_action(self):
        ''' 
        Plots to visualise the best action for a given state 
        '''
        best_actions = []
        for state in self.all_states:
            state_index = self.all_states_dict[state]
            best_action_index = np.argmax(self.q_table[state_index])
            best_action = self.all_actions[best_action_index]
            best_actions.append(best_action)

        plt.plot(self.all_states, best_actions)
        plt.xlabel("State")
        plt.ylabel("Best Action")
        plt.title("Best Action for Each State Q-Learning")
        plt.show()

        return best_actions

#### Create an instance of the class ####
water_tank = WaterTank()

num_episodes = 10000
h0 = 13   # initial height in tank
v0 = 0   # initial outflow
EPSILON = 0.99  # exploration rate
GAMMA = 0.9  # discount factor
ALPHA = 0.5  # learning rate
terminal_state = 10  # desired water level
interval = 0.1
num_states = int((13 - 7) / interval) + 1
all_states = np.linspace(7, 13, num=num_states, endpoint=True)
all_states = np.round(all_states, 1)
all_actions = np.linspace(0, 3.5, 16)
all_states_dict = {state: index for index, state in enumerate(all_states)}
all_actions_dict = {action: index for index, action in enumerate(all_actions)}

# initialise a q-table for use in algorithm
initial_q_values = -50
q_table = np.full((len(all_states), len(all_actions)), initial_q_values)
q_table[all_states_dict[terminal_state]] = 0

q_learning_agent = QLearningAgent(water_tank, all_states, all_actions, all_states_dict, q_table)
q_learning_agent.q_train(num_episodes, h0, v0, EPSILON, GAMMA, ALPHA, terminal_state)

best_actions_list = q_learning_agent.plot_state_vs_best_action()
print(best_actions_list)


######################################################
# Plots to see an averaged policy over multiple runs #
######################################################

num_runs = 1 # how many runs to average
all_runs_best_actions = []

# run the code multiple times and store the best actions for each run
for run in range(num_runs):
    q_learning_agent = QLearningAgent(water_tank, all_states, all_actions, all_states_dict, q_table)
    q_learning_agent.q_train(num_episodes, h0, v0, EPSILON, GAMMA, ALPHA, terminal_state)
    best_actions_list = q_learning_agent.plot_state_vs_best_action()
    all_runs_best_actions.append(best_actions_list)

#### Confidence interval ####
average_best_actions = np.mean(all_runs_best_actions, axis=0)
upper_bound = np.max(all_runs_best_actions, axis=0)
lower_bound = np.min(all_runs_best_actions, axis=0)

plt.figure(figsize=(10, 6))

# Plot the shaded area between upper and lower bounds caused by noise
plt.fill_between(all_states, lower_bound, upper_bound, color='lightblue', alpha=0.5)

# Plot the average line
plt.plot(all_states, average_best_actions, label="Average", linewidth=2, color='blue')

plt.xlabel("State")
plt.ylabel("Best Action")
plt.title("Best Action for Each State Q-Learning")
plt.legend()
plt.grid(False)  # Turn off the grid
plt.show()
