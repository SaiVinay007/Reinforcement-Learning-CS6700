import numpy as np
from matplotlib import pyplot as plt

from pb1 import Epsilon_greedy
from pb2 import Softmax
from pb3 import UCB1

def compare_all(k, steps, runs, true_values, epsilon, temparature, c):
    
    avg_epsilon,_ = Epsilon_greedy(k, epsilon, steps, runs, true_values)
    avg_ucb1,_  =  UCB1(k, steps, runs, true_values, c)
    avg_softmax,_ = Softmax(k, steps, runs, true_values, temparature)
    
    fig=plt.figure(figsize=(10,6)).add_subplot(111)
    
    plt.plot(range(len(avg_epsilon)), avg_epsilon, 'r', label = "Epsilon-greedy: $\epsilon$ = " + str(epsilon) )
    plt.plot(range(len(avg_softmax)), avg_softmax, 'k', label = "Softmax: T = " + str(temparature))
    plt.plot(range(len(avg_ucb1)), avg_ucb1, 'b', label = "UCB1: c = " + str(c))
    
    
    plt.xlabel('Steps', fontsize = 15)
    plt.ylabel('Average reward', fontsize = 15)
    plt.title('Comparition between Epsilon greedy, Softmax, UCB1', fontsize = 15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.ylim(-0.2, 1.6)


    plt.show()
    

if __name__ == '__main__':
    
    k = 10
    steps = 1000
    runs = 2000

    mean = 0
    std_dev = 1
    true_values = np.random.normal(mean, std_dev, (runs, k))

    epsilon = 0.1
    temparature = 0.1
    c = 5
    
    compare_all(k, steps, runs, true_values, epsilon, temparature, c)

if __name__ == '__main__':
    
    k = 1000
    steps = 1000
    runs = 2000

    mean = 0
    std_dev = 1
    true_values = np.random.normal(mean, std_dev, (runs, k))

    epsilon = 0.1
    temparature = 0.1
    c = 2
    
    compare_all(k, steps, runs, true_values, epsilon, temparature, c)