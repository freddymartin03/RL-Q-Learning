# Reinforcement learning  
This repository is to keep track of my work for my Undergraduate Research Opportunity Program (UROP). The code draws upon methods dicussed in 
the book 'Reinforcement Learning: An Introduction' 
by Andrew Barto and Richard S. Sutton. 

## Q-learning (off policy TD control) to a grid-world environment
Q-learning is a powerful off-policy temporal-difference (TD) control algorithm which can be used to outline the optimal action based on the current state of the agent.
Q-learning has shown promising results in various fields, such as robotics, autonomous
systems, and game playing, where finding the optimal action is essential for achieving
specific goals. This report investigates the use of Q-learning in a grid-world environment to develop the optimal policy in which an agent should take to maximise the
cumulative reward.

The image below shows the reward structure for the grid-world environment.

<img width="211" alt="Rewards" src="https://github.com/freddymartin03/RL-UROP/assets/139906764/842266af-482b-4bc4-bf63-673a686d1024">

By updating the rewards table using the Q-learning algorithm (pseudo code below), the agent can learn from its
experiences and improve its performance over successive iterations. Hence, over time, this
allows the agent to differentiate between ’good’ (ones which maximise reward) and ’bad’
(ones which minimise reward) actions in different states, leading to optimal policy learning.

<pre>
Initialize Q(s, a), for all s ∈ S^+, a ∈ A(s), arbitrarily except that Q(terminal, ·) = 0
Loop for each episode:
  Initialize St
  Loop for each step of the episode:   
    Choose At from St using a policy derived from Q (e.g., ε-greedy)
    Take action At, observe Rt+1, St+1
    Q(St, At) ← Q(St, At) + α[Rt+1 + γ maxa Q(St+1, a) − Q(St, At)]
    St ← St+1
  until St is terminal
</pre>

In carrying out the Q-learning algorithm the action-value function, Q, eventually reached
convergence to the optimal action-value function q∗. Hence an optimal policy was developed,
for the given grid-world environment, and is illustrated below. This policy maximises
the expected reward through taking the best action, given the agent is in a certain state.
As highlighted below it is clear to see that the optimal policy is one in which the agent
takes the shortest path from the start cell (the cell in the bottom left) to the terminal end
cell.

<img width="212" alt="Policy" src="https://github.com/freddymartin03/RL-UROP/assets/139906764/f3a36cf2-c6cb-4658-9f3d-91e3427c5bf5">


