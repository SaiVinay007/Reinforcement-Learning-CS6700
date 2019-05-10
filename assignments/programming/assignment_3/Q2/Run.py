import tensorflow as tf
import argparse
import random
import numpy as np
import gym


class DQN():

    
    def __init__(self, args):
        
        # Hyperparameters
        self.hidden_layer1_size = args.h1
        self.hidden_layer2_size = args.h2
        self.hidden_layer3_size = args.h3

        self.epsilon = args.epsilon
        self.epsilon_decay = args.eps_decay
        self.ep_up = args.epsilon_update
        
        self.target_update_freq = args.target_freq
        self.EPISODES = args.episodes
        self.gamma = args.gamma
        self.LAMBDA = args.Lambda
        self.batch_size = args.batch_size


        # State representation size and action space size 
        self.input_size = args.input_size
        self.output_size = args.output_size

        # Input placeholder for obtaining state representation
        self.x = tf.placeholder(dtype = tf.float32, shape= [None, self.input_size], name = 'input')

        # initializing ennvironment
        self.env = gym.make('CartPole-v0')



    def Q_network(self):
        
        ### The primary network

        # Initializing primary networks weights and biases
        # All the weights and biases of primary network
        self.primary_weights = {
            'w1' : tf.Variable(np.random.normal(0, 0.01, size = (self.input_size, self.hidden_layer1_size)), dtype = tf.float32, name = "primary_weight1" ),
            'b1' : tf.Variable(np.random.normal(0, 0.01, size = (self.hidden_layer1_size)), dtype = tf.float32, name = "primary_bias1" ),
            'w2' : tf.Variable(np.random.normal(0, 0.01, size = (self.hidden_layer1_size, self.hidden_layer2_size)), dtype = tf.float32, name = "primary_weight2" ),
            'b2' : tf.Variable(np.random.normal(0, 0.01, size = (self.hidden_layer2_size)), dtype = tf.float32, name = "primary_bias2" ),
            'w3' : tf.Variable(np.random.normal(0, 0.01, size = (self.hidden_layer2_size, self.hidden_layer3_size)), dtype = tf.float32, name = "primary_weight3" ),
            'b3' : tf.Variable(np.random.normal(0, 0.01, size = (self.hidden_layer3_size)), dtype = tf.float32, name = "primary_bias3" )
        }

        # Defining the primary network
        p_h1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.primary_weights['w1']), self.primary_weights['b1']))
        p_h2 = tf.nn.relu(tf.add(tf.matmul(p_h1, self.primary_weights['w2']), self.primary_weights['b2']))

        # Q value from primary network
        self.Q_primary = tf.add(tf.matmul(p_h2, self.primary_weights['w3']), self.primary_weights['b3'])


        ### The target network

        # Initializing target networks weights and biases
        # All the weights and biases of target network
        self.target_weights = {
            'w1' : tf.Variable(np.random.normal(0, 1, size = (self.input_size, self.hidden_layer1_size)), dtype = tf.float32, name = "target_weight1" ),
            'b1' : tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size)), dtype = tf.float32, name = "target_bias1" ),
            'w2' : tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size, self.hidden_layer2_size)), dtype = tf.float32, name = "target_weight2" ),
            'b2' : tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size)), dtype = tf.float32, name = "target_bias2" ),
            'w3' : tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size, self.hidden_layer3_size)), dtype = tf.float32, name = "target_weight3" ),
            'b3' : tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer3_size)), dtype = tf.float32, name = "target_bias3" )
        }

        # Defining the target network
        t_h1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.target_weights['w1']), self.target_weights['b1']))
        t_h2 = tf.nn.relu(tf.add(tf.matmul(t_h1, self.target_weights['w2']), self.target_weights['b2']))
        
        # Q value from target network
        self.Q_target = tf.add(tf.matmul(t_h2, self.target_weights['w3']), self.target_weights['b3'])


        ### Loss and optimizer

        # We need to get the Q values for the actions performed so we multiply the 
        # one hot form of action to the Q values predicted from the primary network
        self.action = tf.placeholder(dtype = tf.int32, shape= [None], name = 'actions') 
        self.one_hot_action = tf.one_hot(self.action, self.output_size, 1.0, 0.0, name = 'one_hot_action')
        self.predicted = tf.reduce_sum(tf.multiply(self.Q_primary, self.one_hot_action, name = 'predicted'), reduction_indices = [1])
        # self.predicted = tf.placeholder(dtype = tf.float32, shape= [None], name = 'predicted')
        
        # We here get the reward + the output of Q values from target network
        self.expected = tf.placeholder(dtype = tf.float32, shape= [None], name = 'expected')

        # The mse loss        
        # self.loss = tf.reduce_mean(tf.square(self.expected - self.predicted)) 
        self.loss = tf.losses.mean_squared_error(self.expected, self.predicted)
        
        # Writing the summary to tensorboard
        tf.summary.scalar('loss', self.loss)
        
        # to keep track of the number of updates made 
        self.global_step_tensor = tf.train.get_or_create_global_step()                                                        

        # optimizer updates the weights so as to decrease the loss, we chose adam optimizer
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)


        

    def train(self, args, flag):

        # stores the average reward
        '''
        avg_rew = tf.Variable(np.zeros(100), dtype = tf.float32, name = "average_total_reward" )
        tf.summary.scalar('average reward', avg_rew)
        merged = tf.summary.merge_all()
        '''
        # creating a session
        self.session = tf.Session()
        
        summary_writer = tf.summary.FileWriter(args.summaries_dir + '/train', self.session.graph)	
        summary = tf.Summary()	


        # Writing summaries
        # train_writer = tf.summary.FileWriter(args.summaries_dir + '/train', self.sess.graph)
        # test_writer = tf.summary.FileWriter(args.summaries_dir + '/test')

        # initializing all the variables
        self.session.run(tf.global_variables_initializer())
        # summary = self.sess.run(merged)                  

        # Initializing replay memory
        self.replay = ReplayMemory(args)
        
        # The total reward obtained in each episode
        total_reward = np.zeros([self.EPISODES])
        # Keeping track of total steps over all episodes
        total_steps = 0

        prev100_episodes = []
        avg_steps = []
        
        # Run for max episodes
        for i in range(self.EPISODES):
            # for every episode we reset the environment
            curr_state = self.env.reset() 
            # number of curr_steps in current episode
            curr_steps = 0

            while True:
                # taking a step
                # self.env.render()
                
                # select an action 
                curr_action = self.select_action(curr_state)
                # get the next state, rewards and a bool telling if episode has terminated or not
                next_state, reward, done, _ = self.env.step(curr_action)

                # increase the steps in current episode, total steps, reward in current episode
                curr_steps+=1
                total_steps+=1
                total_reward[i] += reward

                # add the transition into replay memory
                self.replay.add_experience(curr_state, curr_action, reward, next_state, done)

                # if replay memory has size more than batch, we perform updates to the primary network by 
                # sampling a batch of transitions from replay memory
                if total_steps/self.batch_size > 1 and flag:
                    batch = self.replay.sample_memory()
                    loss_val = self.primary_update(batch)
                    # print("step = ", total_steps , "loss = ", loss_val )
                if not flag:
                    loss_val = self.up_without_mem(curr_state, curr_action, next_state, reward)

                # Updating the target networks weights at some fixed interval to primary network weights : hard update
                if total_steps%self.target_update_freq == 0 :
                    for j in self.primary_weights:
                        self.session.run(tf.assign(self.target_weights[j], self.primary_weights[j]))
                    print("hi =====================================================================================")
                
                # decaying epislon i.e, exploration 
                if total_steps%self.ep_up:
                    self.epsilon *= self.epsilon_decay
                    if self.epsilon < 0.05:  
                        self.epsilon = 0.05
                                    
                # updating current state
                curr_state = next_state
                        
                # Termination 
                if done or curr_steps > 200:
                    break
            
            if (len(prev100_episodes) < 100):
                prev100_episodes.append(curr_steps)
            else:
                prev100_episodes[i % 100] = curr_steps
            mean = np.mean(prev100_episodes)
            avg_steps.append(mean)


            summary.value.add(tag="episode length", simple_value=curr_steps)
            summary.value.add(tag = "Average reward over 100 episode)", simple_value = mean)

            summary_writer.add_summary(summary, i)

            print("Training: Episode = %d, Length = %d, Global step = %d" % (i, curr_steps, total_steps))

            # print(" totalreward = %d", total_reward[i], "curr_steps = ", curr_steps ,"\n")
            
                

    def select_action(self, curr_state):
        # selects action according to epsilon-greedy on Q value function
        if np.random.uniform(0,1)<self.epsilon:
            return np.random.choice([0,1], p=[0.5,0.5])
        else:
            curr_state = np.expand_dims(curr_state, axis = 0)
            Q_values = self.session.run(self.Q_primary, feed_dict={self.x : curr_state})
            # print(Q_values)
            return np.argmax(Q_values) 

    def up_without_mem(self, state, action, next_state, reward):
        # without replay buffer 
        state = np.expand_dims(state, axis = 0)
        next_state = np.expand_dims(state, axis = 0)

        expected_val = self.gamma*np.amax(self.session.run(self.Q_target, feed_dict={self.x : next_state}), axis=1) + reward
        _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict={self.x : state, self.expected : expected_val, self.action : action})

    
    def primary_update(self, batch, flag):
        
        
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

        for i in range(len(states)):
            expected_val = rewards[i]
            expected_val = np.expand_dims(expected_val, axis = 0)
            
            if not terminals[i]:
                next_states[i] = np.expand_dims(next_states[i], axis = 0)
                expected_val = self.gamma*np.amax(self.session.run(self.Q_target, feed_dict={self.x : next_states[i]}), axis=1) + rewards[i] 
            # expected_val = np.expand_dims(expected_val, axis = 0)
            states[i] = np.expand_dims(states[i], axis = 0)
            actions[i] = np.expand_dims(actions[i], axis = 0)

            _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict={self.x : states[i], self.expected : expected_val, self.action : actions[i]})

        return loss_val
            
    
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


    # def test(self):



class ReplayMemory():
    '''
    Creates memory to store, add and get experiences
    for training Q network
    '''

    def __init__(self, args):
        # batch size for sampling from replay memory
        self.batch_size = args.batch_size
        # Initialize replay memory D to capacity N
        self.memory = []
        # max number of experience that can be stored
        self.MAXSIZE = args.memory_size
        # the position at which we add a new experience
        self.position = 0
    

    def add_experience(self, curr_state, curr_action, reward, next_state, done):    
        # Until the memory size is less than max_size
        if len(self.memory) < self.MAXSIZE:
            # Increase to add an experience
            self.memory.append(None)
        # the transition that we want to store
        experience = (curr_state, curr_action, reward, next_state, done)
        # add experience at the current position
        self.memory[self.position] = experience
        # Increase the position by 1
        self.position = (self.position+1)%self.MAXSIZE


    def sample_memory(self):
        # Get a batch size of experiences from the memory 
        batch = random.sample(self.memory, self.batch_size)
        return batch





if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Inputs to the code")

    parser.add_argument("--input_size",     type=float, default=4,            help="input state vector size")
    parser.add_argument("--output_size",    type=float, default=2,            help="output action vector size")

    parser.add_argument("--h1",             type=float, default=64,           help="size of hidden layer1")
    parser.add_argument("--h2",             type=float, default=64,           help="size of hidden layer2")
    parser.add_argument("--h3",             type=float, default=2,            help="size of hidden layer3")

    parser.add_argument("--epsilon",        type=float, default=0.5,          help="epsilon value")
    parser.add_argument("--eps_decay",      type=float, default=0.995,        help="epsilon value")
    parser.add_argument("--epsilon_update", type=float, default=20,           help="epsilon update freq")
    parser.add_argument("--gamma",          type=float, default=0.95,         help="gamma value")
    parser.add_argument("--Lambda",         type=float, default=0.001,        help="regularization lambda value")
    parser.add_argument("--learning_rate",  type=float, default=0.01,        help="learning rate for training")

    parser.add_argument("--target_freq",    type=float, default=300,         help="target network update frequency")
    parser.add_argument("--episodes",       type=float, default=2000,          help="number of episodes for training")
    parser.add_argument("--memory_size",    type=float, default=10000,        help="memory max size of replay memory")
    parser.add_argument("--batch_size",     type=int,   default=20,           help="Batch Size")
    parser.add_argument("--summaries_dir",  type=str,   default='./summary/', help="path to tensorboard summary")



    args = parser.parse_args()

    dqn = DQN(args)
    # initializing Q networks
    dqn.Q_network()
    
    dqn.train(args, False)

    # Visualize the learned behaviour for a few episodes
    results = []
    for i in range(50):
    	episode_length = dqn.playPolicy()
    	print("Test steps = ", episode_length)
    	results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))	    
    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")