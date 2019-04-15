#########################################
############### Problem 6 ############### 
#########################################

import gym
import gym_pdw

import numpy as np
from matplotlib import pyplot as plt


class FA_SARSA_lambda:
    # For updating parameters
    def update(self, curr_state, curr_action, reward, next_state, next_action, E, alpha, gamma, theta):
        feat1 = self.make_feature(curr_state, curr_action)
        feat2 = self.make_feature(next_state, next_action)
        error = reward + (gamma*(np.matmul(np.transpose(theta),feat2))) - (np.matmul(np.transpose(theta),feat1))
        theta = theta + alpha*error*E
        return theta
    
    # For converting to binary formate
    def bin_array(self, num, m):
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
    
    # makes the binary feature of each coordinate,  action and concatenates
    def make_feature(self, state, action):
        x,y = state
        v1 = self.bin_array(x,7)
        v2 = self.bin_array(y,11)
        v3 = np.zeros([4])
        v3[action] = 1
        f = np.concatenate((v1,v2,v3))
        return f
        
    
    def select_action(self, epsilon, state, theta, env):
        if np.random.uniform(0,1) < epsilon:
            action = env.random_action()
        else:
            Q = []
            for i in range(4):
                tmp_feat = self.make_feature(state, i)
                val = np.matmul(np.transpose(theta),tmp_feat)
                Q.append(val)

            action = np.argmax(Q)
        
        return action
    
        
    def fa_sarsa_lambda(self, gamma, alpha, epsilon, episodes, env, lambda_):
        
        # Making a large grid
        new_grid, goal_region = env.large_puddle_world(10, 100, 'A') 
        
        # The boundary coordinates of goal region
        x1,x2,y1,y2 = goal_region
        
        # start positions
        start_pos = env.large_start_pos(new_grid, 10, 100)

        # The parameters
        theta = np.random.rand(22)

        # The num steps and avg_reward that we get from each episode is stored
        steps = np.zeros([episodes])
        avg_reward = np.zeros([episodes])
        
        
        
        for episode in range(episodes):
            # Eligibility
            E = np.zeros([22])
            
            curr_state = env.large_reset(start_pos)
            
            curr_action = self.select_action(epsilon, curr_state, Q, env)            
            # feature of current state,action
            feat = self.make_feature(curr_state, curr_action)
            
            if episode%20 ==0:
                epsilon -= 0.1
                if epsilon <0.1:
                    epsilon = 0.1
            Max_steps = 100000
            for i in range(Max_steps):
                
                E += 1
                next_state, reward = env.large_step(new_grid, curr_state, curr_action)                
                next_action = self.select_action(epsilon, next_state, theta, env)


                E = gamma*lambda_*E + feat
                
                # Updating the parameters
                theta = self.update(curr_state, curr_action, reward, next_state, next_action, E, alpha, gamma, theta)
                
                
                curr_state = next_state
                curr_action = next_action
                
                
                steps[episode]+=1
                avg_reward[episode] += reward


                if (curr_state[0]>=x1  and curr_state[0]<x2 and 
                    curr_state[1]>=y1  and curr_state[1]<y2):
                    # print("Steps =======================", steps[episode])
                    # print("reward=======================", avg_reward[episode])
                    break

        return avg_reward, steps, theta
    
    def plot_fa_sarsa_lambda(self, avg_reward, steps, episodes):
        '''
        Gets the data for all curves and plots them in one graph
        '''


        # Figure instances will be returned.
        fig1=plt.figure(figsize=(10,6)).add_subplot(111)
        fig2=plt.figure(figsize=(10,6)).add_subplot(111)

        # colors for different values of epsilon
        colors = ['g', 'r', 'k', 'b', 'y','m', 'c']

        fig1.plot(range(episodes), avg_reward, colors[0], label = " Average reward " )
        fig2.plot(range(episodes), steps, colors[1], label = " Steps")

        # Labelling the plot
        fig1.title.set_text('Linear funciton approximator SARSA : Avg reward vs episodes')
        fig1.set_ylabel('Average Reward')
        fig1.set_xlabel('episodes')
        fig1.set_ylim(top = 20)
        fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # Labelling the plot
        fig2.title.set_text('Linear funciton approximator SARSA: Num steps vs episodes')
        fig2.set_ylabel('Steps')
        fig2.set_xlabel('episodes')
        fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # Display the plot
        plt.show()

def main():    
    # parameters
    # We here increase gamma as the state space is very large we want to update even when we get reward 
    # at far location i.e, after long time.
    gamma = 1000
    alpha = 0.01
    epsilon = 0.5
    episodes = 100
    
    lambda_ = [0.1, 0.5]
    
    env = gym.make('gym_pdw:pdw-v0')
    fa = FA_SARSA_lambda()
    for i in range(len(lambda_)):
        avg_reward, steps, theta = fa.fa_sarsa_lambda(gamma, alpha, epsilon, episodes, env, lambda_[i])
        fa.plot_fa_sarsa_lambda(avg_reward, steps, episodes)
#     plot_policy(goal_pos,Q)    

main()