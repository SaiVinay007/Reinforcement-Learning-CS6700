import numpy as np
from matplotlib import pyplot as plt

def Softmax(k, steps, runs, true_values, temp):
    
    avg = np.zeros([steps])
    opt = np.zeros([steps])
    
    print(temp)
    opt_arms = np.argmax(true_values, axis=1)
    for i in range(runs):
        
        Q = np.zeros([k])
        N = np.zeros([k])
        
        for j in range(steps):
            softmax = (np.exp(Q/temp))/np.sum(np.exp(Q/temp)) # gets softmax probabilities of all arms
            sf_arm = np.random.choice(range(k), 1, p = softmax ) # picks one arm based on probabilities
            sf_arm = np.squeeze(sf_arm)

            reward = np.random.normal(true_values[i][sf_arm],1)
           
            N[sf_arm] += 1
            Q[sf_arm] = ((Q[sf_arm])*(N[sf_arm]-1) + reward)/N[sf_arm] # average of the values of arm
        
            avg[j]+=reward
#             if j%50==0:
#                 print(reward)
            
            if sf_arm == opt_arms[i]:
                opt[j]+=1
            
            
    avg = np.divide(avg, runs)
    opt = np.divide(opt, runs/100)
    
    return avg, opt
    


def plot_all(avg_reward, opt_percent, temparatures):



    fig1=plt.figure()
    fig2=plt.figure()

    fig1 = fig1.add_axes([0.1, 0.1, 0.6, 0.75])
    fig2 = fig2.add_axes([0.1, 0.1, 0.6, 0.75])

    x = np.zeros([len(avg_reward[0])])
    for i in range(1,steps+1):
        x[i-1] = i

    colors = ['k', 'r', 'b', 'm', 'y','g', 'c']
    for i in range(len(avg_reward)):
        fig1.plot(x, avg_reward[i], colors[i], label = "T = " + str(temparatures[i]) )

    for i in range(len(opt_percent)):
        fig2.plot(x, opt_percent[i], colors[i], label = "T = " + str(temparatures[i]) )

    fig1.title.set_text('Softmax : Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.set_ylim(-0.5,1.6)
    fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig2.title.set_text('Softmax : $\%$ Optimal Action Vs Steps for 10 arms')
    fig2.set_ylabel(r'$\%$ Optimal Action')
    fig2.set_xlabel('Steps')
    fig2.set_ylim(0,100)
    fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig1 = fig1.get_figure()
    fig2 = fig2.get_figure()

    fig1.savefig('sof_re.jpg')
    fig2.savefig('sof_per.jpg')
    plt.show()


if __name__ == '__main__':
    steps = 1000
    runs = 2000
    k = 10
    mean = 0
    std_dev = 1
    
    true_values = np.random.normal(mean, std_dev, (runs, k))
    temparatures = [0.01, 0.1, 1, 10]
    
#     avg, opt = Softmax(k, steps, runs, true_values, temparatures[1])
#     plot_fig(avg, opt)
    avg_reward = []  
    opt_arm = []
    
    colors = ['g', 'r', 'b', 'k', 'y','m', 'c']

    for i in range(len(temparatures)):
        print(i)
        avg, opt = Softmax(k, steps, runs, true_values, temparatures[i])
        avg_reward.append(avg)
        opt_arm.append(opt)
    
    plot_all(avg_reward, opt_arm, temparatures)
        
        