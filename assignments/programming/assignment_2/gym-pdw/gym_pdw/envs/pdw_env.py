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
        # temp = [[6,0],[7,0],[10,0],[11,0]]
        # self.current_position = temp[idx]

        # self.current_position = get_start_positions()[idx]
        

        # actions possible
        # Our origin is on the top left corner
        self.actions = {0 : [-1,0], # North
                        1 : [0,1],  # East
                        2 : [0,-1], # West
                        3 : [1,0]   # South
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
    


    def get_start_positions(self):
        s_p = [[6,0],[7,0],[10,0],[11,0]]
        # print(s_p)
        return s_p


    def set_goal(self,goal):
        if goal=='A':
            x, y = self.goal_positions[0]
            self.grid[x,y] = 10
            self.wind = 1
            return self.goal_positions[0]
        elif goal=='B':
            x, y = self.goal_positions[1]
            self.grid[x,y] = 10
            self.wind = 1
            return self.goal_positions[1]
        elif goal=='C':
            x, y = self.goal_positions[2]
            self.grid[x,y] = 10
            self.wind = 0
            return self.goal_positions[2]


    def get_reward(self, position):
        # The values of matrix contains the reward of transitioning into that state
        self.reward = self.grid[position[0],position[1]]
        return self.reward

    
    # def get_state(self):
    #     return self.current_position


    def actual_action(self, selected_action):
        # Get the probabilities of performing an action 
        # print("********")
        # print(selected_action)
        probs = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
        probs[selected_action] = 0.9
        # print(probs)
        direction = np.random.choice([0,1,2,3],1,p = probs) # if p = is not given, its not working
        direction = direction[0]
        # print(direction)
        # print("********")

        return direction


    # def get_action_probs(self, selected_action):
        
    #     return probs


    def step(self, curr_state, action):
        # Return the postion,reward after performing an action.
        action = self.actual_action(action)

        # Because of wind
        if self.wind:
            self.push = np.random.choice(range(2),1,[0.5,0.5])
            self.push = self.push[0]
        else:
            self.push = 0
        

        if (curr_state[0] + self.actions[action][0] < 0 or
            curr_state[0] + self.actions[action][0] > 11 or
            curr_state[1] + self.actions[action][1] + self.push < 0 or
            curr_state[1] + self.actions[action][1] + self.push > 11)  :

            self.reward = self.get_reward(curr_state)
            next_state = curr_state
            # print("if",self.current_position, self.reward, "Step")
            return next_state, self.reward

        else : 
            x = curr_state[0] + self.actions[action][0]
            y = curr_state[1] + self.actions[action][1] + self.push
            next_state = [x,y]
            self.reward = self.get_reward(next_state)

            return next_state, self.reward


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
        self.pos = s_pos[idx]
        # self.current_position = pos
        # self.grid = self.make_grid()
        return self.pos
        

    def large_puddle_world(self , scale_x, scale_y, goal):
        self.new_grid = np.zeros([12*scale_x,12*scale_y])
        
        # decx = scale_x - 1
        # decy = scale_y - 1

        # The puddle
        self.new_grid[5*scale_x  : 8*scale_x   , 6*scale_y  :8*scale_y ]      -= 1
        self.new_grid[6*scale_x  : 8*scale_x   , 7*scale_y  :8*scale_y ]      += 1
        self.new_grid[4*scale_x  : 9*scale_x   , 5*scale_y  :9*scale_y ]      -= 1
        self.new_grid[7*scale_x  : 9*scale_x   , 8*scale_y  :9*scale_y ]      += 1
        self.new_grid[3*scale_x  : 10*scale_x  , 4*scale_y  :10*scale_y]     -= 1
        self.new_grid[8*scale_x  : 10*scale_x  , 9*scale_y  :10*scale_y]     += 1

        # The Goals
        # Previous goals = [[0,11],[2,9],[7,8]] 
        if goal == 'A':
            self.new_grid[0*scale_x : 1*scale_x , 11*scale_y: 12*scale_y ] = 10
            self.goal_region = [0*scale_x, 1*scale_x, 11*scale_y, 12*scale_y ]

        elif goal =='B':
            self.new_grid[2*scale_x : 3*scale_x , 9*scale_y : 10*scale_y] = 10            
            self.goal_region = [2*scale_x, 3*scale_x, 9*scale_y, 10*scale_y ]

        elif goal=='C':
            self.new_grid[7*scale_x : 8*scale_x , 8*scale_y : 9*scale_y ] = 10          
            self.goal_region = [7*scale_x, 8*scale_x, 8*scale_y, 9*scale_y ]

        return self.new_grid, self.goal_region 

    def large_rewards(self, new_grid):
        self.reward = new_grid[position[0],position[1]]
        return self.reward        

    def large_start_pos(self, new_grid):
        # The start positions  
        # Previous start positions = [[6,0],[7,0],[10,0],[11,0]]
        self.new_start_pos = []
        st_ps = self.get_start_positions()
        for k in range(len(st_ps)):
            x,y = st_ps[k]
            for i in range(x*scale_x,(x+1)*scale_x):
                for j in range(y,(y+1)*scale_y):
                    self.new_start_pos.append([i,j])
        return self.new_start_pos

        
    def large_reset(self, l_start_positions):
        # Initialize the start state
        idx = np.random.choice(range(len(l_start_positions)))
        self.pos = l_start_positions[idx]
        return self.pos
        # self.grid = self.make_grid()

    def large_step(curr_state, action):
        # Return the postion,reward after performing an action.
        action = self.actual_action(action)

        # Because of wind
        if self.wind:
            self.push = np.random.choice(range(2),1,[0.5,0.5])
            self.push = self.push[0]
        else:
            self.push = 0
        

        if (curr_state[0] + self.actions[action][0] < 0 or
            curr_state[0] + self.actions[action][0] > 11 or
            curr_state[1] + self.actions[action][1] + self.push < 0 or
            curr_state[1] + self.actions[action][1] + self.push > 11)  :

            self.reward = self.large_rewards(curr_state)
            next_state = curr_state
            # print("if",self.current_position, self.reward, "Step")
            return next_state, self.reward

        else : 
            x = curr_state[0] + self.actions[action][0]
            y = curr_state[1] + self.actions[action][1] + self.push
            next_state = [x,y]
            self.reward = self.large_rewards(next_state)

            return next_state, self.reward
    
    def render(self, mode='human'):
        ...

    def close(self):
        ...

    
