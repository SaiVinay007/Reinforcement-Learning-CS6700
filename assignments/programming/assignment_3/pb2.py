import tensorflow as tf
import numpy as np
import random

import gym


class DQN():

    def __init__(self):
        
        
        # Initialize action-value function Q with random weights theta

        # Initialize target action-value function Q- weights theta- = theta

        # 

    def primary_network(self):
        # The Q network
        

    def target_network(self):

    



    def select_action(self):
        # selects action according to epsilon-greedy on Q value function

    
    def primary_update(self):


    def target_soft_update(self):
        # Changes the target_network parameters slowly to the primary_network


    def target_hard_update(self):
        # Changes the target_network parameters all at once to the primary_network



class ReplayMemory():
    '''
    Creates memory to store, add and get experiances
    for training Q network
    '''
    def __init__(self):
        # Initialize replay memory D to capacity N
        self.memory = []
        # max number of experiance that can be stored
        self.max_size = 1000
        # the position at which we add a new experiance
        self.position = 0
    
    def add_experiance(self, curr_state, curr_action, reward, next_state, next_action ):    
        # Untill the memory size is less than max_size
        if len(self.memory) < self.max_size:
            # Increase to add an experiance
            self.memory.append(None)
        # the transition that we want to store
        experiance = (curr_state, curr_action, reward, next_state, next_action)
        # add experiance at the current position
        self.memory[self.position] = experiance
        # Increase the position by 1
        self.position = (self.position+1)%self.max_size
    
    def sample_memory(self, batch_size):
        # Get a batch size of experiances from the memory 
        batch = random.sample(self.memory, batch_size)
        return batch


if __name__=='__main__':

    env = gym.make('CartPole-v1')

    inti_dqn = DQN()






