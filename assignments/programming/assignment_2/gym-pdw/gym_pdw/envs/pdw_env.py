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

        # self.start_positions = [[6,0],[7,0],[10,0],[11,0]]

        # Goal positions A, B, C
        self.goal_positions = [[0,11],[2,9],[7,8]]

        # Initialize the start state
        idx = np.random.choice([0,1,2,3])
        # The current position
        temp = [[6,0],[7,0],[10,0],[11,0]]
        self.current_position = temp[idx]

        # self.current_position = get_start_positions()[idx]
        

        # actions possible
        # Our origin is on the top left corner
        self.actions = {0 : [-1,0], # North
                        1 : [0,1], # East
                        2 : [0,-1], # West
                        3 : [1,0] # South
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
    
    # def make_grid(self):
    #     self.grid = np.zeros([12,12], dtype=np.int64)
    #     self.grid[5:8,6:8]      -= 1
    #     self.grid[6:8,7]        += 1
    #     self.grid[4:9,5:9]      -= 1
    #     self.grid[7:9,8]        += 1
    #     self.grid[3:10,4:10]    -= 1
    #     self.grid[8:10,9]       += 1

    #     return self.grid


    def get_start_positions(self):
        s_p = [[6,0],[7,0],[10,0],[11,0]]
        # print(s_p)
        return s_p


    def set_goal(self,goal):
        if goal=='A':
            x, y = self.goal_positions[0]
            self.grid[x,y] += 10
            self.wind = 1
            return self.goal_positions[0]
        elif goal=='B':
            x, y = self.goal_positions[1]
            self.grid[x,y] += 10
            self.wind = 1
            return self.goal_positions[1]
        elif goal=='C':
            x, y = self.goal_positions[2]
            self.grid[x,y] += 10
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
            # print(self.current_position, self.reward, "Step")
            # print("if")
            return self.current_position, self.reward

        else : 
            x = self.current_position[0] + self.actions[self.direction][0]
            y = self.current_position[1] + self.actions[self.direction][1] + self.push
            self.current_position = [x,y]
            self.reward = self.get_reward(self.current_position)
            # print(self.current_position, self.direction, self.reward, "Step")
            # print("else",self.actions[self.direction][0], self.actions[self.direction][1] + self.push )
            return self.current_position, self.reward

        


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
        s_pos = self.get_start_positions()
        # print(s_pos, idx)
        pos = s_pos[idx]
        self.current_position = pos
        # self.grid = self.make_grid()
        

    
    def render(self, mode='human'):
        ...

    def close(self):
        ...

    
