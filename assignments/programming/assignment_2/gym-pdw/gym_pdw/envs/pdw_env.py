import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class PdwEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    
    def __init__(self):

        # Initialize the grid world
        self.grid = np.zeros([12,12], dtype=np.int64)
        
        # Start positions
        self.start_positions = [[6,0],[7,0],[10,0],[11,0]]

        # Goal positions A, B, C
        self.goal_positions = [[0,11],[2,9],[7,8]]

        # Initialize the start state
        idx = np.random.choice([0,1,2,3])
        # The current position
        # self.current_position = self.start_positions[idx]
        

        # actions possible
        # Our origin is on the top left corner
        self.actions = {0 : [-1,0] # North
                        1 : [0,1], # East
                        2 : [0,-1] # West
                        3 : [1,0], # South
                        } 
                        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low = -3, high = 10, shape = self.grid.shape)
        # print(self.action_space)
        # print(self.observation_space)
        # Set rewards
        # For puddle
        self.grid[5:8,6:8]      -= 1
        self.grid[6:8,7]        += 1
        self.grid[4:9,5:9]      -= 1
        self.grid[7:9,8]        += 1
        self.grid[3:10,4:10]    -= 1
        self.grid[8:10,9]       += 1

        # initilalize reward
        self.reward = 0

    def set_goal(self,goal):
        if goal=='A':
            self.grid[self.goal_positions[0]] += 10
            self.wind = 1
            return self.goal_positions[0]
        elif goal=='B':
            self.grid[self.goal_positions[1]] += 10
            self.wind = 1
            return self.goal_positions[1]
        elif goal=='C':
            self.grid[self.goal_positions[2]] += 10
            self.wind = 0
            return self.goal_positions[2]


    def get_reward(self, position):
        # The values of matrix contains the reward of transitioning into that state
        self.reward = self.grid[position[0],position[1]]
        return self.reward

    def get_state(self):
        return self.current_position
      
    def step(self, selected_action):
        # Return the postion,reward after performing an action.

        self.probs = self.get_action_probs(selected_action)
        self.direction = np.random.choice(range(4),1,self.probs)
        self.direction = self.direction[0]

        # Because of wind
        if self.wind:
            self.push = np.random.choice(range(2),1,[0.5,0.5])
            self.push = self.push[0]
        else:
            self.push = 0

        # print(self.push)
        # print(self.direction)


        if (self.current_position[0] + self.actions[self.direction][0] < 0 or
            self.current_position[0] + self.actions[self.direction][0] > 11 or
            self.current_position[1] + self.actions[self.direction][1] + self.push < 0 or
            self.current_position[1] + self.actions[self.direction][1] + self.push > 11)  :

            self.reward = self.get_reward(self.current_position)
            return self.current_position, self.reward

        else : 
            self.current_position[0] += self.actions[self.direction][0]
            self.current_position[1] += self.actions[self.direction][1] + self.push
            
            self.reward = self.get_reward(self.current_position)
            return self.current_position, self.reward

        return

    def get_action_probs(self, selected_action):
        # Get the probabilities of performing an action 
        self.probs = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
        self.probs[selected_action] = 0.9

        return self.probs


    def random_action(self):
        # Pick a random action
        self.action = np.random.choice([0,1,2,3])
        # print(self.action)
        return self.action


    def reset(self):
        # Initialize the start state
        idx = np.random.choice([0,1,2,3])
        print(self.start_positions, idx)
        pos = self.start_positions[idx]
        self.current_position = pos
        print(self.current_position)
        # The current position
        # self.current_position = self.start_position

        # return self.current_position

    
    def render(self, mode='human'):
        ...

    def close(self):
        ...

    
