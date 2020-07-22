#################
#   Problem 3   #
#################

import numpy as np
from matplotlib import pyplot as plt
import math

from pb1 import Epsilon_greedy
from pb2 import Softmax

def UCB1(k, steps, runs, true_values, c):
    '''
    This function performs UCB1 method on 10-armed testbed.
    Parameters :
        k = number of arms
        c = parameter in ucb algorithm
        steps = number of times we pull the arms
        runs = number of bandit problems
        true_values = the actual expected returns of each arm for all bandit problems (size = [runs, k])
    returns :
        avg = the average reward over all bandit problems at each step 
        opt = the % times optimal arm is selected at each step over all bandit problems.
    '''

    '''
    opt_arms stores the index of optimal arm in each of the 2000 bandit problems.
    '''
    opt_arms = np.argmax(true_values,axis = 1)

    '''
    Each value of *avg* stores the average reward over all bandit problems at each step.
    Each value of *opt* stores the number of times optimal arm has been selected over all bandit problems at each step.  
    '''
    avg = np.zeros([steps])
    opt = np.zeros([steps])
    
    print(c)
    
    '''
    For each bandit problem
    '''
    for i in range(runs):       
        '''
        Initialize the expected values of each arm and 
        the nubmer of times each  arm is pulled to zero. 
        '''
        Q = np.zeros([k])
        N = np.zeros([k])
        
              
        '''
        For each step (i.e, pull)
        '''
        for j in range(steps):
            
            '''
            Pull each arm once
            '''    
            if j < k:
                # Sample the reward from distribution, with mean as actual mean of selected arm and standard deviation = 1
                reward = np.random.normal(true_values[i][j], 1)
                # Increase the number of times the arm is pulled by 1                
                N[j] += 1
                # Change the expected value of the selected arm by averaging over all the previous rewards obtained by pulling this arm. 
                Q[j] = Q[j] + (reward - Q[j])/N[j]
                # Increase the number of times optimal arm is pulled at the current step, if optimal arm is selected.            
                if opt_arms[i] == j:
                    opt[j]+=1
                
            
            else:
                '''
                Get the upper confidence bounds of all arms in a bandit problem,
                '''
                upper_bounds = Q + np.sqrt(c*np.log(j)/N)
                
                # Selecting the arm which has highest upper confidence bound
                max_arm = np.argmax(upper_bounds)

                # Sample the reward from distribution, with mean as actual mean of selected arm and standard deviation = 1
                reward = np.random.normal(true_values[i][max_arm],1)

                # Increase the number of times the arm is pulled by 1                
                N[max_arm] += 1

                # Change the expected value of the selected arm by averaging over all the previous rewards obtained by pulling this arm. 
                Q[max_arm] = Q[max_arm] + (reward - Q[max_arm])/N[max_arm]         
                
                # Increase the number of times optimal arm is pulled at the current step, if optimal arm is selected.            
                if opt_arms[i] == max_arm:
                    opt[j]+=1

            # Add the reward obtained at each step.
            avg[j] += reward
               
    # Average of reward over all bandit problems at each step.
    avg = np.divide(avg, runs)
    # percentage of the times that optimal action is chosen over all bandit problems at a time step.
    opt = np.divide(opt, runs/100)
        
    return avg, opt
    
def compare_all(k, steps, runs, true_values, epsilon, temparature, c):

    # Get the average rewards and % optimal actions using Epsilon greedy method 
    avg_epsilon, opt_eps= Epsilon_greedy(k, epsilon, steps, runs, true_values)
    
    # Get the average rewards and % optimal actions using Softmax method 
    avg_softmax, opt_sof = Softmax(k, steps, runs, true_values, temparature)

    # Get the average rewards and % optimal actions using UCB1 method     
    avg_ucb1, opt_ucb1 =  UCB1(k, steps, runs, true_values, c)
    
    
    # Figure instances will be returned.
    fig1=plt.figure(figsize=(10,6)).add_subplot(111)
    fig2=plt.figure(figsize=(10,6)).add_subplot(111)
    
    # Plot the Average reward Vs Steps for Epsilon-greedy, Softmax, UCB 
    fig1.plot(range(steps), avg_epsilon, 'r', label = "Epsilon-greedy: $\epsilon$ = " + str(epsilon) )
    fig1.plot(range(steps), avg_softmax, 'k', label = "Softmax: T = " + str(temparature))
    fig1.plot(range(steps), avg_ucb1, 'g', label = "UCB1: c = " + str(c))
    
    # Plot the %Optimal action Vs Steps for Epsilon-greedy, Softmax, UCB 
    fig2.plot(range(steps), opt_eps, 'r', label = "Epsilon-greedy: $\epsilon$ = " + str(epsilon) )
    fig2.plot(range(steps), opt_sof, 'k', label = "Softmax: T = " + str(temparature))
    fig2.plot(range(steps), opt_ucb1, 'g', label = "UCB1: c = " + str(c))
    
    # Labelling the plot
    fig1.title.set_text(' Average reward comparition between Epsilon greedy, Softmax, UCB1')
    fig1.set_xlabel('Steps', fontsize = 15)
    fig1.set_ylabel('Average reward', fontsize = 15)
    fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Labelling the plot
    fig2.title.set_text('$\%$ Optimal action comparition between Epsilon greedy, Softmax, UCB1')
    fig2.set_xlabel('Steps', fontsize = 15)
    fig2.set_ylabel('$\%$ Optimal Action', fontsize = 15)
    fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Display the plot
    plt.show()


def plot_ucb(avg_reward, opt_arm, C):

    # Figure instances will be returned.
    fig1=plt.figure(figsize=(10,6)).add_subplot(111)
    fig2=plt.figure(figsize=(10,6)).add_subplot(111)

    # colors for different values of c
    colors = ['k', 'r', 'g', 'm', 'y','k', 'c']

    # For each value of c, plot the average reward vs steps
    for i in range(len(avg_reward)):
        fig1.plot(range(steps), avg_reward[i], colors[i], label = "c = " + str(C[i]) )
    
    # For each c, plot the % times optimal arm selected vs steps
    for i in range(len(opt_arm)):
        fig2.plot(range(steps), opt_arm[i], colors[i], label = "c = " + str(C[i]) )
    
    # Labelling the  plot
    fig1.title.set_text('UCB1 : Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.set_ylim(-0.5,1.6)
    fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Labelling the plot
    fig2.title.set_text('UCB1 : $\%$ Optimal Action Vs Steps for 10 arms')
    fig2.set_ylabel(r'$\%$ Optimal Action')
    fig2.set_xlabel('Steps')
    fig2.set_ylim(0,100)
    fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Display the plot
    plt.show()


if __name__ == '__main__':

    '''
    Defining the parameters
    '''
    steps = 1000
    runs = 2000
    k = 10
    mean = 0
    std_dev = 1
    C = [0.1, 2, 5]

    # For storing the "avg", "opt" returned by running UCB1 on 10-armed testbed
    avg_reward = []
    opt_arm = []

    # Initializing the actual expected rewards of each arm, sampled from normal distribution with above mean and standard deviation.
    true_values = np.random.normal(mean, std_dev, (runs, k))
    
    # Running over all c
    for i in range(len(C)):
        avg, opt = UCB1(k, steps, runs, true_values, C[i])
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    plot_ucb(avg_reward, opt_arm, C)


    