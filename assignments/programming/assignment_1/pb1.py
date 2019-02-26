'''
Complete code
'''
import numpy as np
from matplotlib import pyplot as plt


'''
This was a set of 2000 randomly generated k-armed bandit
problems with k = 10
'''


def Epsilon_greedy(k, epsilon, steps, runs, true_values):
    
    opt_arms = np.argmax(true_values,axis=1)
    
    avg = np.zeros([steps])
    opt = np.zeros([steps])
    
    
    for i in range(runs):
        '''
        Initialize the expected values of each action to zero 
        '''
        Q = np.zeros([k])
        N = np.zeros([k])
            
        '''
        Pull the arms 2000 times by following epsilon-greedy approach
        '''
        
        for j in range(steps):
            num = np.random.uniform(0,1)
            
            
            if num<epsilon:
                arm = np.random.randint(0,k)   # pick a random arm
            else:
                arm = np.argmax(Q, axis = 0)    # get the max expectation 
            
            reward = np.random.normal(true_values[i][arm],1)
            N[arm] += 1
            Q[arm] = Q[arm] + (reward - Q[arm])/N[arm]
            if opt_arms[i] == arm:
                opt[j] += 1
        
            avg[j] += reward # To store rewards obtained on every step of every run

    # print(exp_val)
    # print(mx, rn)
    avg = np.divide(avg,runs)  # average of reward at each step over all bandit problems
    opt = np.divide(opt, runs/100)  # percentage of the times that optimal action is chosen at a time step over all bandit problems
    
    return avg,opt


def plot_all(avg_reward, opt_percent):



    fig1=plt.figure()
    fig2=plt.figure()

    fig1 = fig1.add_axes([0.1, 0.1, 0.6, 0.75])
    fig2 = fig2.add_axes([0.1, 0.1, 0.6, 0.75])

    x = np.zeros([len(avg_reward[0])])
    for i in range(1,steps+1):
        x[i-1] = i

    colors = ['g', 'r', 'b', 'k', 'y','m', 'c']
    for i in range(len(avg_reward)):
        fig1.plot(x, avg_reward[i], colors[i], label = "$\epsilon$ = " + str(epsilons[i]) )

    for i in range(len(opt_percent)):
        fig2.plot(x, opt_percent[i], colors[i], label = "$\epsilon$ = " + str(epsilons[i]) )

    fig1.title.set_text(r'$\epsilon$-greedy : Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.set_ylim(-0.5,1.6)
    fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig2.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
    fig2.set_ylabel(r'$\%$ Optimal Action')
    fig2.set_xlabel('Steps')
    fig2.set_ylim(0,100)
    fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig1 = fig1.get_figure()
    fig2 = fig2.get_figure()

    # fig1.savefig('fig1.jpg')
    # fig2.savefig('fig2.jpg')
    plt.show()


if __name__ == '__main__':
    
    steps = 1000
    runs = 2000
    epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    k = 10
    mean = 0
    std_dev = 1

    avg_reward = []  # store for all epsilons
    opt_arm = []     # store for all epsilons
    
    true_values = np.random.normal(mean, std_dev, (runs, k )) 
    
    for i in range(len(epsilons)):
        avg, opt = Epsilon_greedy(k, epsilons[i], steps, runs, true_values)
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    plot_all(avg_reward, opt_arm, epsilons)