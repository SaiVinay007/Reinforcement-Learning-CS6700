import numpy as np
from ads import UserAdvert

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

        # ---- UPDATE code below ------j
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)
        action = int(np.random.choice(range(3)))
        reward = env.getReward(stateId, action)
        # ----------------------------

        # ---- UPDATE code below ------
        # Update policy using reward
        policy = [1/3.0, 1/3.0, 1/3.0]
        # ----------------------------

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
                policy = [1/3.0, 1/3.0, 1/3.0]
                # ----------------------------
                act = int(np.random.choice(range(3), p=policy))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list
    print(rew_vec)


if __name__ == '__main__':
    learnBandit()
