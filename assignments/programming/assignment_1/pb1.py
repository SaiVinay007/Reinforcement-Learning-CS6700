import numpy as np
from matplotlib import pyplot as plt


'''
This was a set of 2000 randomly generated k-armed bandit
problems with k = 10
'''


def bandits(epsilon, steps, runs, mean, std_dev):
    
    avg = np.zeros([steps])
    opt = np.zeros([steps])
    
    
    true_values = np.random.normal(mean, std_dev, (runs, k )) 
    # print(true_values)
    optimal_arm = np.argmax(true_values,axis=1)
    # print(true_values)
    
    for i in range(runs):
        '''
        Initialize the expected values of each action to zero 
        '''
        exp_val = np.zeros([10,2])
            
        '''
        Pull the arms 2000 times by following epsilon-greedy approach
        '''
        
        for j in range(steps):
            num = np.random.uniform(0,1)
            
            if num<epsilon:
                random_arm = np.random.randint(0,10)   # pick a random arm
                reward = np.random.normal(true_values[i][random_arm],1)   # get the reward using the true expectation
                exp_val[random_arm][1] += 1   # increase the count of the arm
                exp_val[random_arm][0] = (exp_val[random_arm][0]*(exp_val[random_arm][1]-1) + reward)/exp_val[random_arm][1]   # average of the values of arm
                if optimal_arm[i] == random_arm:
                    opt[j] += 1
            else:
                max_arm = np.argmax(exp_val, axis = 0)    # get the max expectation 
                reward = np.random.normal(true_values[i][max_arm[0]],1)   # get reward using true distribution
                exp_val[max_arm[0]][1] += 1   # increase the count of arm
                exp_val[max_arm[0]][0] = ((exp_val[max_arm[0]][0])*(exp_val[max_arm[0]][1]-1) + reward)/exp_val[max_arm[0]][1] # average of the values of arm
                # print(max_arm[0], optimal_arm)
                
                if optimal_arm[i] == max_arm[0]:
                    opt[j] += 1
        
            avg[j] = avg[j]+reward # for each step add the reward obtained

    # print(exp_val)
    # print(mx, rn)
    avg = np.divide(avg,runs)
    opt = np.divide(opt, 20.0)
    return avg,opt


def plot_fig(avg_reward, opt_percent):



    fig1=plt.figure()
    fig2=plt.figure()

    fig1 = fig1.add_axes([0.1, 0.1, 0.75, 0.75])
    fig2 = fig2.add_axes([0.1, 0.1, 0.75, 0.75])

    x = np.zeros([steps])
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

    avg_reward = []
    opt_arm = []
    for i in range(len(epsilons)):
        a, b = bandits(epsilons[i], steps, runs, mean, std_dev)
        avg_reward.append(a)
        opt_arm.append(b)
    
    plot_fig(avg_reward, opt_arm)