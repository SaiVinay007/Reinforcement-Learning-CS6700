#################
#   Problem 5   #
#################

import numpy as np
from ads import UserAdvert
from matplotlib import pyplot as plt

ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10


def learnBandit():
    env = UserAdvert()
    rew_vec = []

    
    for train_step in range(TRAIN_STEPS):
        state = env.getState()
        stateVec = state["stateVec"] 
        stateId = state["stateId"]   
        
        # ---- UPDATE code below ------
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)      
        
        ''' Initializing Weight matrix and the update step parameter alpha '''
        if train_step == 0:
            W = np.zeros([4,3])
            alpha = 0.1

        ''' The preferences of each action obtained by multiplying stateVec and W matrices '''
        prefs = np.dot(stateVec, W)     
        
        ''' The policy is obtained by the softmax probabilities each action '''
        policy = np.exp(prefs)/np.sum(np.exp(prefs))           
        
        ''' Selecting an action based on the probabilities obtained '''
        action = int(np.random.choice(range(3), p = policy)) 
        reward = env.getReward(stateId, action)

        # ----------------------------


        # ---- UPDATE code below ------
        # Update policy using reward
        
        ''' Here "i" loops over all the indexes of stateVec which has same length as W.shape[0] '''
        for i in range(W.shape[0]):

            ''' Here "j" loops over all the indexes of columns of W which has length as W.shape[1] '''
            for j in range(W.shape[1]):
                ''' Here we update each element of W matrix based on REINFORCE algorithm  '''
            
                ''' 
                The formula below is obtained by derivating the action selecting policy
                wrt to each parameter using chain rule.

                All the parameters that are involved in preference of the selected arm
                will get update as per the first formula (if condition) and all the remaining 
                will get updated based on the second formula (else condition)
                '''
                if j == action:
                    W[i,j] = W[i,j] + alpha*reward*policy[action]*(1-policy[action])*stateVec[i]
                else:
                    W[i,j] = W[i,j] - alpha*reward*policy[action]*policy[j]*stateVec[i]
        
        # policy = [1/3.0, 1/3.0, 1/3.0]
        # ----------------------------
        # print(policy)
        if train_step % LOG_INTERVAL == 0:
            print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------
                # Policy = function(testV)

                ''' The preferences of actions obtained by multiplying testV and W '''
                prefs = np.dot(testV, W)
                
                ''' The policy is obtained by the softmax probabilities each action '''
                policy = np.exp(prefs)/np.sum(np.exp(prefs))
                # policy = [1/3.0, 1/3.0, 1/3.0]
                # ----------------------------
#                 print(policy)
                act = int(np.random.choice(range(3), p=np.squeeze(policy)))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list

    ''' Figure instances will be returned. '''
    fig1=plt.figure(figsize=(10,6)).add_subplot(111)
    
    ''' Plot the Average reward Vs Time step for the contextual bandit with 3 arms ''' 
    fig1.plot(range(len(rew_vec)), rew_vec, 'r')

    ''' Labelling the  plot '''
    fig1.title.set_text('Contextual : Average Reward Vs Steps for 3 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')

    ''' Displaying the plot '''
    plt.show()

    print(rew_vec)

if __name__ == '__main__':
    learnBandit()


