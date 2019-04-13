#########################################
############### Problem 2 ############### 
#########################################


import gym
import numpy as np
from matplotlib import pyplot as plt

import multiprocessing as mp

# The puddle world Environment 
import gym_pdw


class SARSA:
    
    def update(self, curr_state, curr_action, reward, next_state, next_action, Q, alpha, gamma):
        '''
        This function takes current state, current action, reward obtained next state, next action,
        current Q, parameters : alpha, gamma.
        Performs SARSA update to update action value function Q  
        '''
        error = reward + (gamma*Q[next_action][next_state[0],next_state[1]]) - Q[curr_action][curr_state[0],curr_state[1]]
        Q[curr_action][curr_state[0],curr_state[1]] = Q[curr_action][curr_state[0],curr_state[1]] + alpha*error
        
        return Q
    
    
    def select_action(self,epsilon, state, Q, env):
        '''
        This function is to select action according to Epsilon-greedy wrt Q value function
        '''
        # The exploration part
        if np.random.uniform(0,1) < epsilon:
            action = env.random_action()
        # The expoition part
        else:
            action = np.argmax(Q[:,state[0],state[1]])
        return action
    
    
    def sarsa(self, gamma, alpha, epsilon, episodes, env, goal):
        '''
        This function performs :    initialization of environment,
                                    sets goal position,
                                    run episodes, 
                                    perform actions,
                                    changes state,
                                    make Q function updates
        Returns :   rewards - An array of total reward in each episode
                    steps   - An array of total steps taken in each episode
                    Q       - The Final Q (action value) function

        '''

        # Initialize the action value function to the size of 
        # [number of actions , number grid cells in x direction , number grid cells in y direction]
        Q = np.random.rand(env.action_space.n, env.observation_space.shape[0], env.observation_space.shape[1])

        # Set the goal position
        goal_pos = env.set_goal(goal)

        # Initialize the steps, rewards array to size of number of episodes
        steps = np.zeros([episodes])
        rewards = np.zeros([episodes])

        
        for episode in range(episodes):
            # For each episode

            # Reset the environment, which return the start position
            curr_state = env.reset()
            # Select action based on current state and Q function
            curr_action = self.select_action(epsilon, curr_state, Q, env)

            # Run untill we reach the goal 
            while True:
                
                # Get the next state and reward based on the current state and action perfomed
                next_state, reward = env.step(curr_state, curr_action)

                # Perform next action based on the above next state, Q function using epsilon greedy method
                next_action = self.select_action(epsilon, next_state, Q, env)

                # print('reward == ', reward,"curr_state==", curr_state, "next_state==", next_state, "action==",curr_action)
                
                # Update the Q function based on SARSA update
                Q = self.update(curr_state, curr_action, reward, next_state, next_action, Q, alpha, gamma)
                                
                # Set the next state and next action as current state and current action
                curr_state = next_state
                curr_action = next_action
                
                # Increase the steps made in the episode
                steps[episode]+=1
                # Increase the reward obtained in the episode 
                rewards[episode] += reward

                # If we reach the goal, we break the loop i.e, episode ends
                if curr_state == goal_pos:
                    # print("Steps =======================", steps[episode])
                    # print("reward=======================", avg_reward[episode])
                    break

        # print("-----------One exp done-----------")
        return rewards, steps, Q

        
        def plot_sarsa(self, avg_reward, steps, episodes):
            '''
            Gets the data for all curves and plots them in one graph
            '''


            # Figure instances will be returned.
            fig1=plt.figure(figsize=(10,6)).add_subplot(111)
            fig2=plt.figure(figsize=(10,6)).add_subplot(111)

            # colors for different values of epsilon
            colors = ['g', 'r', 'k', 'b', 'y','m', 'c']

            fig1.plot(range(episodes), avg_reward, colors[0], label = " Average reward " )
            fig2.plot(range(episodes), steps, colors[1], label = " Steps")

            # Labelling the plot
            fig1.title.set_text('SARSA : Average reward at each episode for 50 experiments')
            fig1.set_ylabel('Average Reward')
            fig1.set_xlabel('episodes')
            fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # Labelling the plot
            fig2.title.set_text('SARSA : Average steps at each episode for 50 experiments')
            fig2.set_ylabel('Steps')
            fig2.set_xlabel('episodes')
            fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # Display the plot
            plt.show()


    # To display the obtained policy
    def plot_policy(self, goal_pos, policy):
        plt.rcParams['figure.figsize'] = [7,7]
        fig, ax = plt.subplots()
        # Plot the values of a 2D matrix or array as color-coded image.
        ax.matshow(pol)
        for i in range(12):
            for j in range(12):
                if [j,i] == goal_pos:
                    # Show G at the goal of grid  
                    ax.text(i,j,'G', va='center', ha='center')
                else:
                    # Show the direction at each grid cell according to policy
                    c = int(pol[j,i])
                    direcs = {0:'↑', 1:'➜', 2:'←', 3:'↓' }
                    ax.text(i, j, direcs[c], va='center', ha='center')



if __name__ == '__main__':
    '''
    Defining the parameters
    '''
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    episodes = 500
    
    # Instance of class SARSA
    sr = SARSA()

    # Number of process to run in parallel here 4  
    pool = mp.Pool(processes = 4)
    # Gets all the outputs into 'data' after all processes are finished, we are running total of 50 experiments
    data = [pool.apply(sr.sarsa,args=(gamma, alpha, epsilon, episodes, gym.make('gym_pdw:pdw-v0'),'A')) for i in range(50)] 

    # Store the average reward at each episode
    avg = np.zeros([episodes])
    # Store the number of steps in each episode
    stp = np.zeros([episodes])
    

    # To get the average rewards at each episode and average number of steps in each episode
    for i in range(episodes):
        for j in range(len(data)):
            avg[i] += data[j][0][i]/len(data)
            stp[i] += data[j][1][i]/len(data)

    # To store final policy obtained for 50 experiments 
    policy = np.zeros([12,12])

    # Used to select most chosen action over all experiments 
    val = {0:0, 1:0, 2:0, 3:0}

    # Selects the most times chosen action over all experiments according to policy in each state
    for i in range(12):
        for j in range(12):
            for k in range(len(data)):
                val[np.argmax(data[k][2], axis=0)[i,j]]+=1
            # Get the most chosen action at the current state
            max_ = max(val, key=val.get)
            val = {0:0, 1:0, 2:0, 3:0}
            policy[i,j] = max_
    
    # The goal positions in the environment
    goal_positions = {'A':[0,11],'B':[2,9],'C':[7,8]}
    # A dict to reference goal in a loop
    temp = {0:'A',1:'B',2:'C'}

    # Plot for all goal variants
    for i in range(len(goal_positions)):
        sr.plot_sarsa(avg, stp, episodes)
        sr.plot_policy(goal_positions[temp[i]], policy)