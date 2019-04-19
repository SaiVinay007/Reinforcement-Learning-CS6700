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

        self.target_update_freq = args.target_freq
        self.tau = args.tau
        self.epsilon = args.epsilon


        # State representation size and action space size 
        self.input_size = args.input_size
        self.output_size = args.output_size

        # Input placeholder for obtaining state representation
        self.x = tf.placeholder(dtype = tf.float32, shape= [None, self.input_size], name = 'input')

        # # Initialize Weights and bias of primary network 
        # self.primary_weights = {

        #                     }

        # # Initialize Weights and bias of target network 
        # self.target_weights = {

        #                     }



    #     # Initialize action-value function Q with random weights theta

    #     # Initialize target action-value function Q- weights theta- = theta

    #     # 

    def Q_network(self):
        
        ## The primary network

        # Initializing primary networks weights and biases
        self.w1 = tf.Variable(np.random.normal(0, 1, size = (self.input_size, self.hidden_layer1_size)), dtype = tf.float32, name = "weight1" )
        self.b1 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size)), dtype = tf.float32, name = "bias1" )
        self.w2 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size, self.hidden_layer2_size)), dtype = tf.float32, name = "weight2" )
        self.b2 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size)), dtype = tf.float32, name = "bias2" )
        self.w3 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size, self.hidden_layer3_size)), dtype = tf.float32, name = "weight3" )
        self.b3 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer3_size)), dtype = tf.float32, name = "bias3" )

        # All the weights and biases of primary network
        self.p_weights = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3] 

        p_h1 = tf.matmul(self.x, self.w1) + self.b1
        p_h1 = tf.nn.relu(p_h1)
        p_h2 = tf.matmul(p_h1, self.w2) + self.b2
        p_h2 = tf.nn.relu(p_h2)

        # Q value from primary network
        self.Q_primary = tf.matmul(p_h2, self.w3) + self.b3


        ## The target network

        # Initializing target networks weights and biases
        self.W1 = tf.Variable(np.random.normal(0, 1, size = (self.input_size, self.hidden_layer1_size)), dtype = tf.float32, name = "weight1" )
        self.B1 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size)), dtype = tf.float32, name = "bias1" )
        self.W2 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer1_size, self.hidden_layer2_size)), dtype = tf.float32, name = "weight2" )
        self.B2 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size)), dtype = tf.float32, name = "bias2" )
        self.W3 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer2_size, self.hidden_layer3_size)), dtype = tf.float32, name = "weight3" )
        self.B3 = tf.Variable(np.random.normal(0, 1, size = (self.hidden_layer3_size)), dtype = tf.float32, name = "bias3" )

        # All the weights and biases of target network
        self.t_weights = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3] 


        t_h1 = tf.matmul(self.x, self.W1) + self.B1
        t_h1 = tf.nn.relu(t_h1)
        t_h2 = tf.matmul(t_h1, self.W2) + self.B2
        t_h2 = tf.nn.relu(t_h2)

        # Q value from target network
        self.Q_target = tf.matmul(t_h3, self.W3) + self.B3
        



    def select_action(self, curr_state):
        # selects action according to epsilon-greedy on Q value function
        if np.random.uniform(0,1)<self.epsilon:
            return np.random.choice([0,1], p=[0.5,0.5])
        else:
            curr_state = np.expand_dims(curr_state,0)
            Q_values = self.sess.run(self.Q_primary, feed_dict={self.x : curr_state})
            print(Q_values)
            return np.argmax(Q_values) 
        

    def train(self, args):

        # initializing ennvironment
        env = gym.make('CartPole-v0')

        # stores the average reward
        avg_rew = tf.Variable(np.zeros(100), dtype = tf.float32, name = "average_total_reward" )
        tf.summary.scalar('average reward', avg_rew)

        merged = tf.summary.merge_all()
        
        self.sess = tf.Session()

        train_writer = tf.summary.FileWriter(args.summaries_dir + '/train', self.sess.graph)
        test_writer = tf.summary.FileWriter(args.summaries_dir + '/test')


        self.sess.run(tf.global_variables_initializer())

        curr_state = env.reset() 
        curr_action = self.select_action(curr_state)
        print("curr",curr_action)
        # # running for 100 episodes
        # for i in range(100):
        #     # 50 experiments to take average
        #     for j in range(50):
                
                

                
                


    
    # def primary_update(self):


    # def target_soft_update(self):
    #     # Changes the target_network parameters slowly to the primary_network


    # def target_hard_update(self):
    #     # Changes the target_network parameters all at once to the primary_network




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
    

    def add_experience(self, curr_state, curr_action, reward, next_state, next_action):    
        # Until the memory size is less than max_size
        if len(self.memory) < self.MAXSIZE:
            # Increase to add an experience
            self.memory.append(None)
        # the transition that we want to store
        experience = (curr_state, curr_action, reward, next_state, next_action)
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

    parser.add_argument("--input_record_file",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/train.tfrecords',help="path to TFRecord file with training examples")
    parser.add_argument("--validation_record_file",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/test.tfrecords',help="path to TFRecord file with test examples")
    parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
    parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to save checkpoints")
    parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to load checkpoints from")
    parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
    parser.add_argument("--display_step",type = int,default=1,help="display frequency")
    parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")
    parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")

    parser.add_argument("--batch_size",type=int,default=8,help="Batch Size")
    parser.add_argument("--summaries_dir",type=str,default='./summary/',help="path to tensorboard summary")
    parser.add_argument("--input_size", type=float, default=4, help="input state vector size")
    parser.add_argument("--output_size", type=float, default=2, help="output action vector size")
    parser.add_argument("--h1", type=float, default=128, help="size of hidden layer1")
    parser.add_argument("--h2", type=float, default=128, help="size of hidden layer2")
    parser.add_argument("--h3", type=float, default=4, help="size of hidden layer3")
    parser.add_argument("--tau", type=float, default=0.05, help="size of hidden layer3")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon value")
    parser.add_argument("--target_freq", type=float, default=50, help="target update frequency")



    args = parser.parse_args()





    env = gym.make('CartPole-v0')
    print(args.batch_size)
    # for i in range(10):
    #     observation = env.reset()
    #     obv = env.observation_space.shape[0]
    #     env.render()
    #     print(observation, obv)
    #     break

    dqn = DQN(args)
    # initializing Q networks
    dqn.Q_network()
    
    dqn.train(args)




