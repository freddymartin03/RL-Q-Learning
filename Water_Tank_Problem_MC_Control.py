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
        The agents action is flowrate in v_in
        '''
        t = np.linspace(t, t+1, 2)
        v_out_uncertainty = np.random.normal(0, 0.5, 1)
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
        After that the episode ends 
        '''
        if t >= 30:
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
# MC learning agent #
####################

class MonteCarloAgent(WaterTank):
    def __init__(self, water_tank, all_states, all_actions, all_states_dict, terminal_state, EPSILON):
        super().__init__()
        self.water_tank = water_tank
        self.all_states = all_states
        self.all_actions = all_actions
        self.all_states_dict = all_states_dict
        self.terminal_state = terminal_state
        self.q_table_mc = np.full((len(all_states), len(all_actions)), 0)
        self.q_table_mc[all_states_dict[terminal_state]] = 0
        self.EPSILON = EPSILON
        self.current_state = None


    def initial_state(self):
        '''
        Provides a starting state for the agent
        '''
        return random.choice(self.all_states.tolist())

    def get_next_state(self, state, action, t):
        '''
        Given the action the agent takes this function returns the next state
        Use of ODE solver to generate next state
        '''
        max_state = np.max(self.all_states)
        min_state = np.min(self.all_states)
        next_state = self.water_tank.ode_solver(state, action, t)

        if np.isnan(next_state):
            next_state = state

        if next_state > max_state:
            next_state = max_state
        elif next_state < min_state:
            next_state = min_state

        self.current_state = next_state
        return next_state

    def behavior_policy(self):
        '''
        Policy b, epsilon-greedy policy
        '''
        if random.random() < self.EPSILON:
            action = random.choice(self.all_actions.tolist())
        else:
            state_index = self.all_states_dict[self.current_state]
            best_action_index = np.argmax(self.q_table_mc[state_index])
            action = self.all_actions[best_action_index]

        return action
    
    def tables(self):
        '''
        Initialises the q-table 
        Initialises the c-table: used to keep track of the
        number of times a specific state-action pair has 
        been visited or encountered during the learning process
        '''
        initial_q_table_mc = -300
        q_table_mc = np.full((len(all_states), len(all_actions)), initial_q_table_mc)
        q_table_mc[all_states_dict[terminal_state]] = 0

        c_table_mc = np.zeros((len(all_states), len(all_actions)))

        targetpolicy_table = np.zeros((len(all_states), len(all_actions)))
        targetpolicy_table = self.policy_improvement(q_table_mc, policy_table=targetpolicy_table)

        return q_table_mc, c_table_mc, targetpolicy_table

    def policy_improvement(self, q_table_mc, policy_table):
        '''
        Updates the q-table
        '''
        for state in self.all_states:
            state_index = self.all_states_dict[state]
            best_action_index = np.argmax(q_table_mc[state_index])
            best_action = self.all_actions[best_action_index]
            policy_table[state_index] = best_action

        return policy_table


    def episode_gen(self, behavior_policy):
        '''
        Generate an episode using the specified behavior policy
        '''
        episode_states = []
        episode_actions = []
        episode_rewards = []

        self.current_state = self.initial_state()  # set the initial state
        t = 0
        while not self.episode_end(t):
            t += 1
            action = behavior_policy()
            next_state = self.get_next_state(self.current_state, action, t)
            reward = self.reward_gen(action, next_state)
            episode_states.append(self.current_state)  
            episode_actions.append(action)
            episode_rewards.append(reward)
            self.current_state = next_state  # update the current state for the next step
        
        return episode_states, episode_actions, episode_rewards

    def target_policy(self, state, action):
        '''
        Compute the probability of selecting the target policy's action given the current states
        '''
        state_index = self.all_states_dict[state]
        best_action_index = np.argmax(self.q_table_mc[state_index])
        best_action = self.all_actions[best_action_index]

        return 1.0 if action == best_action else 0.0

    def off_policy_mc_prediction(self, num_episodes, gamma):
        '''
        Perform off-policy Monte Carlo prediction for estimating action-values
        '''
        Q, C, _ = self.tables()
        state_episode_counts = np.zeros(len(self.all_states))
        for episode in range(num_episodes):
            episode_states, episode_actions, episode_rewards = self.episode_gen(self.behavior_policy)

            G = 0
            W = 1
            for t in range(len(episode_states) - 1, -1, -1):
                state = episode_states[t]
                action = episode_actions[t]
                state_index = self.all_states_dict[state]
                action_index = np.where(self.all_actions == action)[0][0]  # find the index of the action
                state_episode_counts[state_index] += 1
                reward = episode_rewards[t + 1] if t + 1 < len(episode_rewards) else 0


                G = gamma * G + reward
                C[state_index, action_index] += W
                delta = 1e-6  # prevent division by zero
                if not np.isnan(Q[state_index, action_index]):
                    Q[state_index, action_index] += (W / (C[state_index, action_index] + delta)) * (
                                G - Q[state_index, action_index])

                # update importance sampling weight
                W *= self.target_policy(state, action) / (self.behavior_policy() + delta)

            print(f'\rProgress: {episode / num_episodes * 100} %', end='')

        return Q, C, state_episode_counts


#### Create an instance of the class ####
water_tank = WaterTank()

num_episodes = 10000
h0 = 7  # round(random.uniform(7, 13), 1)   initial height in tank
v0 = 0  # np.random.normal(0, 0.5, 1)   initial outflow
EPSILON = 0.99  # exploration rate
GAMMA = 1  # discount factor
ALPHA = 0.5  # learning rate
terminal_state = 10
interval = 0.1
num_states = int((13 - 7) / interval) + 1
all_states = np.linspace(7, 13, num=num_states, endpoint=True)
all_states = np.round(all_states, 1)
all_actions = np.linspace(0, 3.5, 16)
all_states_dict = {state: index for index, state in enumerate(all_states)}
all_actions_dict = {action: index for index, action in enumerate(all_actions)}

mc_agent = MonteCarloAgent(water_tank, all_states, all_actions, all_states_dict, terminal_state, EPSILON)
q_table_mc, c_table_mc, state_episode_counts = mc_agent.off_policy_mc_prediction(num_episodes, GAMMA)
print(q_table_mc)

best_action_list = []
for state in all_states:
    state_index = all_states_dict[state]
    action_index = np.argmax(q_table_mc[state_index])
    best_action = all_actions[action_index]
    best_action_list.append(best_action)
    print(f"State {state:.1f}: Best Action = {best_action:.2f}")
print(best_action_list)

plt.plot(all_states, best_action_list)
plt.xlabel("State")
plt.ylabel("Best Action")
plt.title("Best Action for Each State MC")
plt.show()






