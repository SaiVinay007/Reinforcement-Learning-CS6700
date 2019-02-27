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

    W = np.zeros([4,3])
    alpha = 0.1
    for train_step in range(TRAIN_STEPS):
        state = env.getState()
        stateVec = state["stateVec"] # the representation of the person given to us
        stateId = state["stateId"]   # the bandit that idealy must be selected
        
        # ---- UPDATE code below ------j
        
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)           
        '''
        Here we are learning to select an action based on the state that we are in as per the policy.
        The state here is the person that visits our site.
        
        Softmax says the probability of selecting an action based on the state that we are in.
        '''
        
        prefs = np.dot(stateVec, W)                          # The preferences of each action
        policy = np.exp(prefs)/np.sum(np.exp(prefs))           # The probabilities of selecting each arm
        action = int(np.random.choice(range(3), p = policy)) # add probability of taking each action
        reward = env.getReward(stateId, action)
        # ----------------------------

        # ---- UPDATE code below ------
        # Update policy using reward
        '''
        One possibility maybe pseudo inverse.
        Here we dont no the true reward, so we cannot calculate loss but by expectation we can expect what 
        might we the reward so we differentiate that and update weights.
        '''
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
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
                prefs = np.dot(testV, W)
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
    fig1=plt.figure(figsize=(10,6)).add_subplot(111)
    
    fig1.plot(range(len(rew_vec)), rew_vec, 'r')

    fig1.title.set_text('Contextual : Average Reward Vs Steps for 3 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')

    fig1 = fig1.get_figure()
    fig1.savefig('Contextual.jpg')

#     fig1.set_ylim(-0.5,1.6)
#     fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    print(rew_vec)
    plt.show()

if __name__ == '__main__':
    learnBandit()


