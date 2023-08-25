# Reinforcement learning  
This repository is to keep track of my work for my Undergraduate Research Opportunity Program (UROP). The code draws upon methods dicussed in 
the book 'Reinforcement Learning: An Introduction' 
by Andrew Barto and Richard S. Sutton. [1] 

## Q-learning (off policy TD control) in a grid-world environment
Q-learning is a powerful off-policy temporal-difference (TD) control algorithm which can be used to outline the optimal action based on the current state of the agent.
Q-learning has shown promising results in various fields, such as robotics, autonomous
systems, and game playing, where finding the optimal action is essential for achieving
specific goals. This report investigates the use of Q-learning in a grid-world environment to develop the optimal policy in which an agent should take to maximise the
cumulative reward.

The image below shows the reward structure for the grid-world environment.

<img width="211" alt="Rewards" src="https://github.com/freddymartin03/RL-UROP/assets/139906764/842266af-482b-4bc4-bf63-673a686d1024">

By updating the rewards table using the Q-learning algorithm (pseudo code below), the agent can learn from its
experiences and improve its performance over successive iterations [1]. Hence, over time, this
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

During the learning process, the agent explores the environment by taking actions according to an exploration policy. This allows the agent to discover different state-action pairs.After taking an action and observing the resulting state and reward, the agent updates the Q-value for the current state-action pair using the Q-learning update rule. The figure below shows the TD against the number of episodes. It is interesting to note that the TD appears to settle fastest initially for a lower epsilon value. This can be explained through the fact that the agent chooses to exploit rather than explore, thus few states are encountered. As the number of episodes increases the agent discovers a large reward causing a large update in the TD value, hence the spikes are observed in the figure below. In contrast, a high epsilon value settles faster as the state space is explored rather than exploited thus the TD settles quickly, as all states have been encountered.  

<img width="881" alt="TD_graph" src="https://github.com/freddymartin03/Reinforcement-learning-UROP/assets/139906764/3c5a875f-af50-47ec-aac9-25ce3f9b38b8">

## Q-learning (off policy TD control) Water Tank
To further experiment with Q-learning, I applied the algorithm to a water tank problem. The premise of the problem revolves around a water tank (see below) where Q-learning is used to control the height of water in the tank where the outflow is subject to randomness. In establishing this problem, the tank dynamics are modelled and solved using the SciPy odeint solver. The action that the agent can take is adding water to the tank in order to keep the level at 10m, it is worth noting that the agent can only add water, it is not able to take water out of the tank.  

<img width="400" alt="Tank" src="https://github.com/freddymartin03/Reinforcement-learning-UROP/assets/139906764/79609f11-f201-49dd-85a9-0f8fdcc4642f">


Through applying the Q-learning algorithm discussed earlier a policy can be developed by the agent, in order to achieve, and subsequently maintain the tank level at 10m (ie a state of 10). A graph is provided below which illustrates the policy developed through averaging multiple runs. This approach was taken due to the stochastic nature of the problem in which the agent can experience an abnormally large outflow and hence require a large update of the Q function to develop an optimal policy. The figure below illustrates an averaged policy in which the agent aims to keep the water level at 10m. It is intresting to note that the agent takes actions less than 1.5m^3s^(-1) to reduce the magnitude of the negative reward. 


<img width="712" alt="Q_average_e0 99_n50000_error" src="https://github.com/freddymartin03/Reinforcement-learning-UROP/assets/139906764/a7c611e4-34de-4788-8978-35893a43d7ec">

It is interesting to observe the confidence interval in the figure above. When the agent observes the stochastic nature of the outflow it becomes harder to differentiate between genuinely effective actions and those that only appear effective due to random fluctuations. This can lead to the agent making incorrect value estimations and policy decisions.

## Off-policy Monte Carlo Control
Monte Carlo learning is a powerful technique in the field of reinforcement learning that enables agents to learn optimal strategies through experience and exploration. This approach hinges on the idea of estimating the value of states or state-action pairs by averaging the cumulative rewards obtained from multiple simulated episodes. By collecting and analysing data from the interactions between an agent and an environment, Monte Carlo methods offer a practical solution to decision-making problems across various domains. Through simulating and analysing numerous random trajectories to estimate values, policies, and rewards, Monte Carlo enables the design of effective control policies that adapt to real-world scenarios. The algorithm discussed in the Andrew Barto and Richard S. Sutton book: 'Reinforcement Learning: An Introduction' is shown below [1]. 

<pre>
Input: an arbitrary target policy π
Initialise, for all s ∈ S, a ∈ A(s):
	Q(s, a) ∈ R (arbitrarily)
	C(s, a) ← 0
Loop forever (for each episode):
	b ← any policy with coverage π
	Generate an episode following b: S_0,A_0,R_1,…  ,S_(T-1),A_(T-1),R_T  
	G ← 0
	W ← 1
	Loop for each step of episode, t=T-1,T-2,…  ,0 while W≠0:
                G ← γG + R_(t+1)
		C(S_t,A_t) ←  C(S_t,A_t)+W
		Q(S_t,A_t) ← Q(S_t,A_t)+  W/(C(S_t,A_t)) [G- Q(S_t,A_t)]
		W ← W (π(A_t|S_t))/(b(A_t|S_t))  
</pre>

In the context of the water tank problem, Monte Carlo learning provides a framework to address the challenge of controlling the water level in a tank subject to uncertain outflows. By modelling the system dynamics and simulating episodes, the agent can iteratively learn and refine policies that minimise water level deviations and optimise control actions. By repeatedly sampling and learning from these trajectories, Monte Carlo learning can provide insights into how to make informed decisions to achieve the desired control objective of maintaining the water tank height at 10m, while considering uncertainties.


[1] Sutton, R.S. and Barto, A. (2018). Reinforcement learning : an introduction. Cambridge, Ma ; Lodon: The Mit Press.

‌
