#################
#   Problem 1   #
#################

import numpy as np
from matplotlib import pyplot as plt


def Epsilon_greedy(k, epsilon, steps, runs, true_values):
    '''
    This function performs epsilon greedy method on 10-armed testbed.
    Parameters :
        k = number of arms
        epsilon = the value of parameter epsilon
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
    opt_arms = np.argmax(true_values,axis=1)
    
    '''
    Each value of *avg* stores the average reward over all bandit problems at each step.
    Each value of *opt* stores the number of times optimal arm has been selected over all bandit problems at each step.  
    '''
    avg = np.zeros([steps])
    opt = np.zeros([steps])
    
    print(epsilon)
    
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
            # Get a random number between 0 and 1
            num = np.random.uniform(0,1)
            
            '''
            The condition based on which we select an arm.
                If the above "num" is less than epsilon (probability of happening = epsilon) :
                    we pull a random arm.
                If the above "num" is greater than epsilon (probability of happening = 1 - epsilon) :
                    we pull the arm with highest expectation. 

            '''
            if num<epsilon:
                # A random arm
                arm = np.random.randint(0,k)   
            else:
                # The arm with highest expectation
                arm = np.argmax(Q, axis = 0)   
            
            # Sample the reward from distribution, with mean as actual mean of selected arm and standard deviation = 1
            reward = np.random.normal(true_values[i][arm],1)

            # Increase the number of times the arm is pulled by 1
            N[arm] += 1
            
            # Change the expected value of the selected arm by averaging over all the previous rewards obtained by pulling this arm. 
            Q[arm] = Q[arm] + (reward - Q[arm])/N[arm]

            # Increase the number of times optimal arm is pulled at the current step, if optimal arm is selected.
            if opt_arms[i] == arm:
                opt[j] += 1

            # Add the reward obtained at each step.
            avg[j] += reward 


    # Average of reward over all bandit problems at each step.
    avg = np.divide(avg,runs)  
    # percentage of the times that optimal action is chosen over all bandit problems at a time step.
    opt = np.divide(opt, runs/100)  
    
    return avg,opt


def plot_epsilon(avg_reward, opt_arm, epsilons):
    '''
    Gets the data for all curves and plots them in one graph
    '''

    
    # Figure instances will be returned.
    fig1=plt.figure(figsize=(10,6)).add_subplot(111)
    fig2=plt.figure(figsize=(10,6)).add_subplot(111)

    # colors for different values of epsilon
    colors = ['g', 'r', 'k', 'b', 'y','m', 'c']

    # For each value of epsilon, plot the average reward vs steps
    for i in range(len(avg_reward)):
        fig1.plot(range(steps), avg_reward[i], colors[i], label = "$\epsilon$ = " + str(epsilons[i]) )
    
    # For each epsilon, plot the % times optimal arm selected vs steps
    for i in range(len(opt_arm)):
        fig2.plot(range(steps), opt_arm[i], colors[i], label = "$\epsilon$ = " + str(epsilons[i]) )

    # Labelling the plot
    fig1.title.set_text(r'$\epsilon$-greedy : Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.set_ylim(-0.5,1.6)
    fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Labelling the plot
    fig2.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
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
    # epsilons = [0, 0.01, 0.1]
    epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    k = 10
    mean = 0
    std_dev = 1

    # For storing the "avg", "opt" returned by running Epsilon_greedy on 10-armed testbed
    avg_reward = []  
    opt_arm = []     
    
    # Initializing the actual expected rewards of each arm, sampled from normal distribution with above mean and standard deviation.
    true_values = np.random.normal(mean, std_dev, (runs, k)) 
    
    # Running over all epsilons
    for i in range(len(epsilons)):
        # Getting the average rewards and % optimal actions.
        avg, opt = Epsilon_greedy(k, epsilons[i], steps, runs, true_values)
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    # Displaying all the curves on one graph
    plot_epsilon(avg_reward, opt_arm, epsilons)