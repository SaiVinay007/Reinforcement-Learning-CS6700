#################
#   Problem 4   #
#################

import numpy as np
from matplotlib import pyplot as plt

from pb1 import Epsilon_greedy, plot_epsilon
from pb2 import Softmax, plot_softmax
from pb3 import UCB1, compare_all ,plot_ucb


if __name__ == '__main__':
    
    k = 1000
    steps = 10000
    runs = 2000

    mean = 0
    std_dev = 1
    true_values = np.random.normal(mean, std_dev, (runs, k))
    
    epsilons = [0 ,0.01, 0.1]
    for i in range(len(epsilons)):
        # Getting the average rewards and % optimal actions.
        avg, opt = Epsilon_greedy(k, epsilons[i], steps, runs, true_values)
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    plot_epsilon(avg_reward, opt_arm, epsilons)


    temparatures = [0.01, 0.1, 1, 10]
    for i in range(len(temparatures)):
        # Getting the average rewards and % optimal actions.
        avg, opt = Softmax(k, steps, runs, true_values, temparatures[i])
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    C = [0.1, 2, 5]
    for i in range(len(C)):
        avg, opt = UCB1(k, steps, runs, true_values, C[i])
        avg_reward.append(avg)
        opt_arm.append(opt)

    epsilon = 0.1
    temparature = 0.1
    c = 5    
    compare_all(k, steps, runs, true_values, epsilon, temparature, c)
    


