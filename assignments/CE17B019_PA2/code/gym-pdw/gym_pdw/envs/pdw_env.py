import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class PdwEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    
    def __init__(self):

        # Initialize the grid world
        self.grid = np.zeros([12,12], dtype=np.int64)
        

        # self.start_positions = [[6,0],[7,0],[10,0],[11,0]]

        # Goal positions A, B, C
        self.goal_positions = [[0,11],[2,9],[7,8]]

        # Allowed actions
        # Our origin is on the top left corner
        self.actions = {0 : [-1,0], # North
                        1 : [0,1],  # East
                        2 : [0,-1], # West
                        3 : [1,0]   # South
                        } 

        # All actions
        self.action_space = spaces.Discrete(len(self.actions))
        # All states
        self.observation_space = spaces.Box(low = -3, high = 10, shape = self.grid.shape)
        
        # print(self.action_space)
        # print(self.observation_space)

        # Set rewards for the puddle
        self.grid[5:8,6:8]      -= 1
        self.grid[6:8,7]        += 1
        self.grid[4:9,5:9]      -= 1
        self.grid[7:9,8]        += 1
        self.grid[3:10,4:10]    -= 1
        self.grid[8:10,9]       += 1

        # initilalize reward
        self.reward = 0
    

    # Return possible start positions
    def get_start_positions(self):
        s_p = [[6,0],[7,0],[10,0],[11,0]]
        # print(s_p)
        return s_p

    # Return the the goal position and enables wind for goals : A, B and disable wind for goal : C
    # Also sets the reward at the goal position
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


    # Returns the reward for being in the current state 
    def get_reward(self, position):
        # The values of matrix contains the reward of transitioning into that state
        self.reward = self.grid[position[0],position[1]]
        return self.reward

    
    # def get_state(self):
    #     return self.current_position

    # Returns the action after considering the stochastic nature of actions to take place 
    def actual_action(self, selected_action):
        # Set the probabilities of performing an action 
        probs = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
        probs[selected_action] = 0.9

        # Select an action according to probabilities        
        direction = np.random.choice([0,1,2,3],1,p = probs) # if p = is not given, its not working
        direction = direction[0]

        return direction


    def step(self, curr_state, action):
        # Return the postion,reward after performing an action.

        # Select the action by considering stochastic nature after selecting an action
        action = self.actual_action(action)


        if self.wind:
            # Westerly blowing, that will push you one additional cell to the east with probability of 0.5
            self.push = np.random.choice(range(2),1,[0.5,0.5])
            self.push = self.push[0]
        else:
            self.push = 0
        
        # According to the action taken, returns the reward and the next state
        if (curr_state[0] + self.actions[action][0] < 0 or
            curr_state[0] + self.actions[action][0] > 11 or
            curr_state[1] + self.actions[action][1] + self.push < 0 or
            curr_state[1] + self.actions[action][1] + self.push > 11)  :
            # Transitions that take you off the grid will not result in any change

            self.reward = self.get_reward(curr_state)
            next_state = curr_state
            return next_state, self.reward

        else : 
            x = curr_state[0] + self.actions[action][0]
            y = curr_state[1] + self.actions[action][1] + self.push
            next_state = [x,y]
            self.reward = self.get_reward(next_state)

            return next_state, self.reward


    # Picks a random action
    def random_action(self):
        self.action = np.random.choice([0,1,2,3])
        return self.action

    # Brings our agent back to one of the start state
    def reset(self):
        # select a random start state
        idx = np.random.choice([0,1,2,3])
        s_pos = self.get_start_positions()

        self.pos = s_pos[idx]
        return self.pos
        

    def large_puddle_world(self , scale_x, scale_y, goal):
        '''
        This is for the problem 6
        Here we make a grid of size [120, 1200 ], which is 1000 times enlarged version of previous grid
        '''

        # Initializing the grid
        self.new_grid = np.zeros([12*scale_x,12*scale_y])
        
        # decx = scale_x - 1
        # decy = scale_y - 1

        # Assigning rewards to the grid
        self.new_grid[5*scale_x  : 8*scale_x   , 6*scale_y  :8*scale_y ]      -= 1
        self.new_grid[6*scale_x  : 8*scale_x   , 7*scale_y  :8*scale_y ]      += 1
        self.new_grid[4*scale_x  : 9*scale_x   , 5*scale_y  :9*scale_y ]      -= 1
        self.new_grid[7*scale_x  : 9*scale_x   , 8*scale_y  :9*scale_y ]      += 1
        self.new_grid[3*scale_x  : 10*scale_x  , 4*scale_y  :10*scale_y]      -= 1
        self.new_grid[8*scale_x  : 10*scale_x  , 9*scale_y  :10*scale_y]      += 1

        # The Goals
        # Setting the rewards and wind according to the goal positions
        # Returns the modified grid and the region of goal
        # Previous goal positions [[0,11],[2,9],[7,8]], we scale them accordingly to get new goal region
        if goal == 'A':
            self.new_grid[0*scale_x : 1*scale_x , 11*scale_y: 12*scale_y ] = 10
            self.goal_region = [0*scale_x, 1*scale_x, 11*scale_y, 12*scale_y ]
            self.wind = 1

        elif goal =='B':
            self.new_grid[2*scale_x : 3*scale_x , 9*scale_y : 10*scale_y] = 10            
            self.goal_region = [2*scale_x, 3*scale_x, 9*scale_y, 10*scale_y ]
            self.wind = 1

        elif goal=='C':
            self.new_grid[7*scale_x : 8*scale_x , 8*scale_y : 9*scale_y ] = 10          
            self.goal_region = [7*scale_x, 8*scale_x, 8*scale_y, 9*scale_y ]
            self.wind = 0

        return self.new_grid, self.goal_region 


    # Here we return the reward obtained in a state
    def large_rewards(self, new_grid, state):
        self.reward = new_grid[state[0], state[1]]
        return self.reward        

    # Here we return the possible start position of the new grid
    def large_start_pos(self, new_grid, scale_x, scale_y):
        
        # Previous start positions = [[6,0],[7,0],[10,0],[11,0]]
        self.new_start_pos = []

        # Getting the start positions used in the previous case
        st_ps = self.get_start_positions()

        for k in range(len(st_ps)):
            x,y = st_ps[k]
            for i in range(x*scale_x,(x+1)*scale_x):
                for j in range(y,(y+1)*scale_y):
                    # Appending all the new start positions
                    self.new_start_pos.append([i,j])
        return self.new_start_pos


    # We return to one of the start postion        
    def large_reset(self, l_start_positions):
        # Initialize the start state
        idx = np.random.choice(range(len(l_start_positions)))
        self.pos = l_start_positions[idx]
        return self.pos

    # We make a state transition and reward and new state is returned
    def large_step(self, new_grid, curr_state, action):
        # Return the postion,reward after performing an action.
        action = self.actual_action(action)

        # Because of wind effect
        if self.wind:
            self.push = np.random.choice(range(2),1,[0.5,0.5])
            self.push = self.push[0]
        else:
            self.push = 0

        
        if (curr_state[0] + self.actions[action][0] < 0 or
            curr_state[0] + self.actions[action][0] > 119 or
            curr_state[1] + self.actions[action][1] + self.push < 0 or
            curr_state[1] + self.actions[action][1] + self.push > 1199)  :
            # Transitions that take you off the grid will not result in any change
            
            self.reward = self.large_rewards(new_grid, curr_state)
            next_state = curr_state

            return next_state, self.reward

        else : 
            x = curr_state[0] + self.actions[action][0]
            y = curr_state[1] + self.actions[action][1] + self.push
            next_state = [x,y]
            self.reward = self.large_rewards(new_grid, next_state)

            return next_state, self.reward
    

    def render(self, mode='human'):
        ...

    def close(self):
        ...

    
