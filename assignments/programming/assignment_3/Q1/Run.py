
import gym
import gym_four
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
'''
To change in step() of grid world
'''

class SMDP_Q():

    def __init__(self, env, episodes, intra):

        self.env = env

        self.gamma = 0.9
        self.alpha = 0.1
        self.episodes = episodes
        self.epsilon = 0.1

        self.Q = np.zeros([6, 11, 11])
        self.intra_option = intra

        # print(self.env.action_space)
    
    def smdp_Q(self, env):
        
        # Reset environment to get the current state
        
        steps = np.zeros([self.episodes])
        rewards = np.zeros([self.episodes])

        for episode in range(self.episodes):
            # start state for each episode
            
            state = self.env.reset()
            # state = [8,2]
            while True:

                # Perform an option. Each state has 6 options, out of which 2 are hallway options and 4 are primitive actions
                option = self.select_option(state)
                if option>=4:
                    target_doorway = self.get_target_door(state, option) 
                else:
                    target_doorway = None
                # print(state, option, target_doorway)
                # perform a step
                next_state, reward, done, k = self.perform_option(state, option, target_doorway)
                # print(option, state[0], state[1])
                # print(self.Q.shape)
                # Update the Q value funtion
                if not self.intra_option :
                    self.Q[option][state[0], state[1]] = self.Q[option][state[0], state[1]] + self.alpha*( reward + (self.gamma**k)*np.amax(self.Q[:,next_state[0],next_state[1]]) - self.Q[option][state[0],state[1]] )

                print("state  =", state, "option = ", option,"next_state  =", next_state, "reward = ", reward)

                state = next_state
                steps[episode] += 1
                rewards[episode]+=reward

                if done:
                    # if the goal state is reached
                    print("episode = ", episode, "steps = ", steps[episode], "reward = ",rewards[episode])
                    break
        return  steps, rewards, self.Q


    # Update Q(st; o) using Q-learning update
    def select_option(self, state):
        if np.random.uniform(0,1) < self.epsilon:
            # 0,1,2,3 are primitive actions 4,5 are hallway options 4th option doorway selects doorway 
            # on clockwise to the room and 5th option anticlockwise doorway to the room 
            option = np.random.choice([0,1,2,3,4,5])
        else:
            option = np.argmax(self.Q[:,state[0],state[1]], axis=0)
        return option


    # an option corresponds to a hallway state for a room 
    def perform_option(self, state, option, target_doorway):
        steps = 0
        if option < 4:
            # Set the probabilities of performing an option 
            probs = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
            probs[option] = 0.9
            # Select an option according to probabilities        
            option = np.random.choice([0,1,2,3],1,p = probs) # if "p =" is not given, its not working
            option = option[0]
            next_state, total_reward, done, _ = env.step(state, option, target_doorway)
            steps +=1
            if self.intra_option:
                self.Q[option][state[0], state[1]] = (1.0 - self.alpha)*self.Q[option][state[0], state[1]] + self.alpha*(total_reward + self.gamma * np.amax(self.Q[:,next_state[0],next_state[1]]))

                    
        else: 

            total_reward = 0
            while state != target_doorway:
                steps+=1
                action = self.option_policy(state, target_doorway)
                next_state, reward, done, terminate = self.env.step(state, action, target_doorway)
                total_reward += reward
                state = next_state

                if self.intra_option:
                    # update the Q value using the same option
                    if not terminate:
                        self.Q[option][state[0], state[1]] = (1.0 - self.alpha)*self.Q[option][state[0], state[1]] + self.alpha*(reward + self.gamma * self.Q[option,next_state[0],next_state[1]]) 
                    # update using the max option possible in that state
                    else:
                        self.Q[option][state[0], state[1]] = (1.0 - self.alpha)*self.Q[option][state[0], state[1]] + self.alpha*(reward + self.gamma * np.amax(self.Q[:,next_state[0],next_state[1]]))

                                     
                    

                if(terminate or done):
                    break

        return next_state, total_reward, done, steps



    def get_target_door(self, state, option):
        # Returns the target doorway of the multistep option chosen

        d1,d2 = self.env.doorways[1]
        d3,d4 = self.env.doorways[3]
        option -=4
        
        if state == d1 and option==0:
            target_doorway = self.env.doorways[4][0]
        elif state == d1 and option==1:
            target_doorway = self.env.doorways[1][1]
        elif state == d2 and option==0:
            target_doorway = self.env.doorways[1][0]
        elif state == d2 and option==1:
            target_doorway = self.env.doorways[2][1]
        elif state == d3 and option==0:
            target_doorway = self.env.doorways[2][0]
        elif state == d3 and option==1:
            target_doorway = self.env.doorways[3][1]
        elif state == d4 and option==0:
            target_doorway = self.env.doorways[3][0]    
        elif state == d4 and option==1:
            target_doorway = self.env.doorways[4][1]    
        else :
            room = self.env.get_room(state)
            target_doorway = self.env.doorways[room][option]
        
        return target_doorway
    

    def option_policy(self, state, target_doorway):
        x1,y1 = state
        x2,y2 = target_doorway
        drwys = self.env.get_doorways(state)

        state, flag = self.env.in_doorway(state)
        
        if x2>x1:
            x = 3
        elif x1>x2:
            x = 0
        if y2>y1:
            y = 1
        elif y1>y2:
            y = 2
        
        if flag:
            # if we are in doorway => constrained in one direction
            x_ = self.env.get_reward([x1+self.env.actions[x][0], y1]) 
            Y_ = self.env.get_reward([y1+self.env.actions[y][1], x1])
            if (x_<0):
                action = y
            else:
                action = x

        else:
            # if we are inside the room
            if x1==x2:
                action = y
            elif y1==y2:
                action = x
            else:
                if (abs(x2-x1)>abs(y2-y1) and (self.env.get_reward([state[0]+self.env.actions[x][0],y1])>=0)):
                    action = x
                elif (abs(x2-x1)<abs(y2-y1) and (self.env.get_reward([x1,state[1]+self.env.actions[y][1]])>=0)):
                    action = y            
                else:
                    if (self.env.get_reward([state[0]+self.env.actions[x][0],y1])>=0):
                        action = x
                    else :
                        action = y

        return action
 


    def plot_four(self, avg_reward, steps, episodes):
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
        fig1.title.set_text('SMDP Q learning : Average reward at each episode for 50 experiments for goal  ')
        fig1.set_ylabel('Average Reward')
        fig1.set_xlabel('episodes')
        # fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # Labelling the plot
        fig2.title.set_text('SMDP Q learning : Average steps at each episode for 50 experiments ' )
        fig2.set_ylabel('Steps')
        fig2.set_xlabel('episodes')
        # fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # Display the plot
    plt.show()
    

if __name__=='__main__':

    env = gym.make('gym_four:four-v0')
    episodes = 1000
    intra = False
    obj = SMDP_Q(env, episodes, intra)
    runs = 50
    # Store the average reward at each episode
    avg = np.zeros([episodes])
    # Store the number of steps in each episode
    stp = np.zeros([episodes])

    q = np.zeros([11,11])

    for i in range(runs):
        steps, rewards, Q = obj.smdp_Q(env)
        q+=np.sum(Q, axis=0)
        stp+= steps/runs
        avg+= rewards/runs

    q = np.sum(Q, axis=0)

    obj.plot_four(avg, stp, episodes)
    plt.show()
 
    ax = sns.heatmap(q, linewidth=0.5, cmap="YlGnBu")
    plt.show()
 
 


