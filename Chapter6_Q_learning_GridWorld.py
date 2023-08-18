import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

environment_rows = 4  # no. of rows in the environment
environment_columns = 4  # no. of columns in the environment
num_actions = 4  # actions are: up, right, down, left
EPSILON = 0.9  # probability of choosing to explore in epsilon-greedy algorithm
GAMMA = 0.9  # discount factor
ALPHA = 0.5  # learning rate for q learning
num_episodes = 500

rewards = np.full((environment_rows, environment_columns), -1)
goal_state = (0, environment_columns - 1)  # goal state will always be the top right corner
rewards[goal_state] = 100

start_state = (environment_rows - 1, 0)  # start state will always be the bottom left corner
rewards[start_state] = -1.

for col in range(1, environment_columns):
    non_state = (environment_rows - 1, col)
    rewards[non_state] = 0


# this function is simply for visualising where the non-states occur
def convert_to_non_state(reward):
    if reward == 0:
        return 'non_state'
    else:
        return str(reward)


rewards_with_non_state = np.vectorize(convert_to_non_state)(rewards)
print(rewards_with_non_state)
print('\n')

# sets the states that have a reward = 0 as not accessible to the agent
valid_states = [(row, col) for row in range(environment_rows) for col in range(environment_columns)
                if rewards[row, col] != 0]
q_values = np.zeros((len(valid_states), num_actions))
actions = ['up', 'right', 'down', 'left']  # outlines the actions we are able to take


# epsilon greedy algorithm that will choose the next action
def next_action(state_index, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(num_actions)
        else:
            return np.argmax(q_values[state_index])


def get_next_location(row_index, column_index, action_index):
    new_row_index = row_index
    new_column_index = column_index
    if actions[action_index] == 'up' and row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and column_index > 0:
        new_column_index -= 1

    # ensure the new state is not a non-state
    if rewards[new_row_index, new_column_index] == 0:
        new_row_index = row_index
        new_column_index = column_index

    # Return the reward for the new state, which is -1 in this case
    return new_row_index, new_column_index


def is_terminal_state(row_index, column_index):
    return (row_index, column_index) == goal_state

# initialize Q-values for the goal state to zero
goal_state_index = valid_states.index(goal_state)
q_values[goal_state_index] = np.zeros(num_actions)

# training the agent
temporal_difference_list = []
num_episodes_list = []
for episode in range(num_episodes):
    row_index, column_index = start_state

    while not is_terminal_state(row_index, column_index):
        state_index = valid_states.index((row_index, column_index))
        action_index = next_action(state_index, EPSILON)

        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        new_state_index = valid_states.index((row_index, column_index))
        reward = rewards[row_index, column_index]

        old_q_value = q_values[state_index, action_index]
        temporal_difference = reward + (GAMMA * np.max(q_values[new_state_index])) - old_q_value

        new_q_value = old_q_value + (ALPHA * temporal_difference)  # Q-learning equation
        q_values[state_index, action_index] = new_q_value

    temporal_difference_list.append(abs(temporal_difference))
    num_episodes_list.append(episode)

plt.plot(num_episodes_list, temporal_difference_list)
plt.xlabel('Number of Episodes')
plt.ylabel('Absolute Value of Temporal Difference')
plt.title('Absolute Value of Temporal Difference vs Number of Episodes' + '\n' 'for Different Epsilon Values')
plt.show()



# adding the Q-values to a dataframe to visualise best actions in a state
q_values_df = pd.DataFrame(q_values,
                           columns=['up', 'right', 'down', 'left'],
                           index=[f'State {i+1}' for i in range(len(valid_states))])


# the below code is for visualizing the result through plotting appropriate tables
max_actions = q_values_df.apply(lambda row: [row == row.max()], axis=1)
q_values_df['Best_actions'] = max_actions
print(q_values_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.axis('off')
table = ax1.table(cellText=rewards_with_non_state, cellLoc='center', loc='center')
table.scale(1, 1.5)
cell_width = 1.2
cell_height = 4.8
table.scale(cell_width, cell_height)
ax1.set_title('Rewards Table', fontsize=18)

ax2.axis('off')


def add_arrows(actions_str, row_index):
    arrow_length = 0.25
    (y, x) = np.divmod(row_index, environment_columns)
    y = (environment_rows - 1) - y  # invert y-coordinate to start from the top

    if (x, y) == (environment_columns - 1, environment_rows - 1):
        ax2.text(x + 0.5, y + 0.5, "End", fontsize=14, ha='center', va='center', color='red')
    else:
        for action_index, is_max in enumerate(actions_str):
            if is_max:
                if actions[action_index] == 'up':
                    ax2.arrow(x + 0.5, y + 0.5, 0, arrow_length, head_width=0.1, head_length=0.1, color='red')
                elif actions[action_index] == 'down':
                    ax2.arrow(x + 0.5, y + 0.5, 0, -arrow_length, head_width=0.1, head_length=0.1, color='red')
                elif actions[action_index] == 'right':
                    ax2.arrow(x + 0.5, y + 0.5, arrow_length, 0, head_width=0.1, head_length=0.1, color='red')
                elif actions[action_index] == 'left':
                    ax2.arrow(x + 0.5, y + 0.5, -arrow_length, 0, head_width=0.1, head_length=0.1, color='red')

    if (x, y) == (0, 0):
        ax2.text(x + 0.5, y + 0.4, "Start", fontsize=14, ha='center', va='center', color='red')


for i, actions_str in enumerate(max_actions):
    add_arrows(actions_str[0], i)


plt.show()




