import gym
import random

import numpy as np
import tensorflow as tf

class DQN:
	
	REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay  
	EPSILON = 0.5 						# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.99 				# exponential decay multiplier for epsilon
	EPSILON_DECAY_FREQ = 40 			# epsilon decay frequency
	HIDDEN1_SIZE = 24					# size of hidden layer 1
	HIDDEN2_SIZE = 24					# size of hidden layer 2
	EPISODES_NUM = 2000 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 					# maximum number of steps in an episode 
	LEARNING_RATE = 0.001 				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 20					# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.95 				# MDP's gamma
	TARGET_UPDATE_FREQ = 1000 			# number of steps (not episodes) after which to update the target networks 
	LOG_DIR = './logs' 					# directory wherein logging takes place
	

	# Create and initialize the environment
	def __init__(self, env):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
	
	# Create the Q-network
	def initialize_network(self):

  		# placeholder for the state-space input to the q-network
		self.x = tf.placeholder(tf.float32, [None, self.input_size])

		############################################################
		# Design your q-network here.
		# 
		# Add hidden layers and the output layer. For instance:
		# 
		# with tf.name_scope('output'):
		#	W_n = tf.Variable(
		# 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size], 
		# 			 stddev=0.01), name='W_n')
		# 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
		# 	self.Q = tf.matmul(h_n-1, W_n) + b_n
		#
		#############################################################

		# Your code here
		with tf.name_scope('output'):
			

			self.weights = {
            	'w1' : tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], mean = 0, stddev=0.1), dtype = tf.float32, name = "weight1" ),
            	'b1' : tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name = "bias1" ),
            	'w2' : tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], mean = 0, stddev=0.01), name = "weight2" ),
            	'b2' : tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name = "bias2" ),
            	'w3' : tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], mean = 0, stddev=0.01), name = "weight3" ),
            	'b3' : tf.Variable(tf.zeros(self.output_size), name = "bias3" )
        	}

			# Defining the primary network
			p_h1 = tf.nn.relu(tf.matmul(self.x, self.weights['w1']) +  self.weights['b1'])
			p_h2 = tf.nn.relu(tf.matmul(p_h1, self.weights['w2']) + self.weights['b2'])

        	# Q value from primary network
			self.Q_primary = tf.matmul(p_h2, self.weights['w3']) + self.weights['b3']
			
			self.target_weights = {
            	'w1' : tf.Variable(tf.truncated_normal( [self.input_size, self.HIDDEN1_SIZE], mean = 0, stddev=0.01), dtype = tf.float32, name = "target_weight1" ),
            	'b1' : tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name = "target_bias1" ),
            	'w2' : tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], mean = 0, stddev=0.01), name = "target_weight2" ),
            	'b2' : tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name = "target_bias2" ),
            	'w3' : tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], mean = 0, stddev=0.01), name = "target_weight3" ),
            	'b3' : tf.Variable(tf.zeros(self.output_size), name = "target_bias3")
        	}
			# Defining the target network
			t_h1 = tf.matmul(self.x, self.target_weights['w1']) + self.target_weights['b1']
			t_h2 = tf.matmul(t_h1, self.target_weights['w2']) + self.target_weights['b2']			
			
			# Q value from target network
			self.Q_target = tf.matmul(t_h2, self.target_weights['w3']) + self.target_weights['b3']

			# self.Q = tf.matmul(p_h2, W_3) + b_3


		############################################################
		# Next, compute the loss.
		#
		# First, compute the q-values. Note that you need to calculate these
		# for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
		#
		# Next, compute the l2 loss between these estimated q-values and 
		# the target (which is computed using the frozen target network)
		#
		############################################################

		# Your code here
		self.replay_buffer = []
		self.position = 0

		self.sel_action = tf.placeholder(dtype = tf.int32, shape= [None], name = 'actions') 
		self.one_hot_action = tf.one_hot(self.sel_action, self.output_size, 1.0, 0.0, name = 'one_hot_action')
		self.predicted = tf.reduce_sum(tf.multiply(self.Q_primary, self.one_hot_action, name = 'predicted'), reduction_indices = [1])

		self.expected = tf.placeholder(dtype = tf.float32, shape= [None], name = 'expected')		
		
		self.loss = tf.losses.mean_squared_error(self.expected, self.predicted)

		############################################################
		# Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam. 
		#
		# For instance:
		# optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
		# global_step = tf.Variable(0, name='global_step', trainable=False)
		# self.train_op = optimizer.minimize(self.loss, global_step=global_step)
		#
		############################################################

		# Your code here
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)


		############################################################

	def train(self, episodes_num=EPISODES_NUM):
		
		# Initialize summary for TensorBoard 						
		summary_writer = tf.summary.FileWriter(self.LOG_DIR)	
		summary = tf.Summary()	
		# Alternatively, you could use animated real-time plots from matplotlib 
		# (https://stackoverflow.com/a/24228275/3284912)
		
		# Initialize the TF session
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################

		# Your code here

		############################################################
		# Main training loop
		# 
		# In each episode, 
		#	pick the action for the given state, 
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################
        
		total_steps = 0
		for j in self.weights:
			self.session.run(tf.assign(self.target_weights[j], self.weights[j]))
		
		
		for episode in range(episodes_num):
		  
			state = self.env.reset()
			state = np.reshape(state, [1, self.input_size])

			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			episode_length = 0
			episode_reward = 0
			#
			############################################################

			while True:

				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################

				# Your code here
				if np.random.uniform(0,1)<self.EPSILON:
					# state = np.expand_dims(state, axis = 0)
					action = np.random.choice([0,1], p=[0.5,0.5])
				else:
					# state = np.expand_dims(state, axis = 0)
					Q_values = self.session.run(self.Q_primary, feed_dict={self.x : state})
					action = np.argmax(Q_values)
					# print(action)
				

				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				# Your code here
				next_state, reward, done, _ = self.env.step(action)
				next_state = np.reshape(next_state, [1, self.input_size])

				episode_length += 1 
				episode_reward += 1
				total_steps += 1 


				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################

				# Your code here
				if len(self.replay_buffer) < self.REPLAY_MEMORY_SIZE:
        		    # Increase to add an experience
					self.replay_buffer.append(None)
				# the transition that we want to store
				experience = (state, action, reward, next_state, done)
				# add experience at the current position
				self.replay_buffer[self.position] = experience
				# Increase the position by 1
				self.position = (self.position+1)%self.REPLAY_MEMORY_SIZE

				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s') 
				#
				# Remember, the target (r + gamma * max Q) is computed    
				# with the help of the target network.
				# Compute this target and pass it to the network for computing 
				# and minimizing the loss with the current estimates
				#
				############################################################

				# Your code here
				if len(self.replay_buffer) >= self.MINIBATCH_SIZE:
					batch = random.sample(self.replay_buffer, self.MINIBATCH_SIZE)

					# all the states of the selected batch
					states = [tup[0] for tup in batch]
					# all the actions of the selected batch
					actions = [tup[1] for tup in batch]
					# all the next states of the selected batch obtained from the previous states
					next_states = [tup[3] for tup in batch]
					# the rewards obtained during this transition
					rewards = [tup[2] for tup in batch]
					# to check for terminal states
					terminals = [tup[4] for tup in batch]
					# print(len(self.replay_buffer))

					for i in range(len(states)):
						expected_val = rewards[i]
						expected_val = np.expand_dims(expected_val, axis = 0)

						if not terminals[i]:
							# next_states[i] = np.expand_dims(next_states[i], axis = 0)
							expected_val = self.DISCOUNT_FACTOR*np.amax(self.session.run(self.Q_target, feed_dict={self.x : next_states[i]}), axis=1) + rewards[i] 
							# states[i] = np.expand_dims(states[i], axis = 0)
							actions[i] = np.expand_dims(actions[i], axis = 0)
							_, loss_val = self.session.run([self.train_op, self.loss], feed_dict={self.x : states[i], self.expected : expected_val, self.sel_action : actions[i]})
							# print("Loss = %d", (loss_val))

                # updating current state
				state = next_state
				
				############################################################
			  	# Update target weights. 
			  	#
			  	# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################

				# Your code here
				if total_steps % self.TARGET_UPDATE_FREQ == 0:
					for j in self.weights:
						self.session.run(tf.assign(self.target_weights[j], self.weights[j]))
				
				
				# decaying epislon i.e, exploration 
				if total_steps%self.EPSILON_DECAY_FREQ == 0:
					# print('--------')
					self.EPSILON *= self.EPSILON_DECAY
					if self.EPSILON < 0.01:  
						self.EPSILON = 0.01

				
				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################
				
				# Your code here
				if done or (episode_length == self.MAX_STEPS):
					break


			############################################################
			# Logging. 
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :

			print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_length, total_steps))
			summary.value.add(tag="episode length", simple_value=episode_length)
			summary_writer.add_summary(summary, episode)


	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200: 	
			self.env.render()				
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps


if __name__ == '__main__':

	# Create and initialize the model
	dqn = DQN('CartPole-v0')
	dqn.initialize_network()

	print("\nStarting training...\n")
	dqn.train()
	print("\nFinished training...\nCheck out some demonstrations\n")

	# Visualize the learned behaviour for a few episodes
	results = []
	for i in range(50):
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)
	print("Mean steps = ", sum(results) / len(results))	

	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")
