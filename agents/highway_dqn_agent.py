import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model, losses, optimizers


class DQNModel(Model):
    def __init__(self, obs_size, action_size):
        super(DQNModel, self).__init__()
        
        self.conv1 = Conv2D(32, kernel_size=(8,8), strides=4, activation='relu')
        self.conv2 = Conv2D(64, kernel_size=(4,4), strides=2, activation='relu')
        self.conv3 = Conv2D(64, kernel_size=(3,3), strides=4, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class DQNAgent():
    def __init__(self, env, savedir="my_model"):
        
        # env, state, and network model
        self.env = env
        self.t = 1
        self.state = env.reset()
        self.obs_size = env.observation_space.shape
        self.act_size = env.action_space.n
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.optimizer = optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
#         self.model.compile(loss='mse', 
#                            optimizer=optimizers.Adam(lr=self.lr))
        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        
        
        # discount parameter
        self.gamma = 0.985
        
        # exploration parameters
        self.random_end_t = 5000
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.1
        
        # threshold parameters for terminating training (1st loop)
        self.max_episodes = 1000
        self.reward_threshold = 40
        self.num_episodes = 1
        
        # step / episode parameters (2nd loop)
        self.episode_t = 1 # reset in each episode
        self.replay_start_t = 100
        self.max_t_per_episode = 1000
        self.episode_reward = 0.0
        
        self.train_period = 4
        self.target_update_period = 1000        
      
        
        # replay buffer and batch parameters
        self.buffer_size = 5000
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 32
     
        
#         # log data
#         self.mean_episode_rewards = []
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = Input(shape=(*self.obs_size,))
        
        layer1 = Flatten()(inputs)
        
        layer2 = Dense(1024, activation="relu")(layer1)
        layer3 = Dense(256, activation="relu")(layer2)
        layer4 = Dense(64, activation="relu")(layer3)
        
        outputs = Dense(self.act_size, activation="linear")(layer4)

        return Model(inputs=inputs, outputs=outputs)

    
    def get_action(self, state, test=True):
        """
        return the optimal action in the given state.
        if "mode" is "train", use the 
        """
        if test:
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
            return tf.argmax(self.model(state_tensor, training=False)[0]).numpy()
        
        if not test:
            if self.t < self.random_end_t or self.epsilon > np.random.random():
                return self.env.action_space.sample()
            else:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
                return tf.argmax(self.model(state_tensor, 
                                            training=False)[0]).numpy()
        
    def step(self, action):
        """
        step in environment for data collection and return doneness
        """
        state = self.state
        next_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward 
        self.buffer.append([state, action, reward, next_state, float(done)])
        self.state = next_state
        return done
   
    def reset_episode(self):
        print("episode reward of {} with {} steps".format(self.episode_reward, 
                                                          self.episode_t))
        
        self.episode_reward = 0.0
        self.episode_t = 1
        self.state = self.env.reset()
        
    def save_model(self, filename="my_model"):
        self.model.save(filename)
        
    def load_model(self, filename="my_model"):
        self.model = tf.keras.models.load_model(filename)
                               
    def sample(self, batch_size):
        """
        sample a minibatch from a buffer
        """
        return random.sample(self.buffer, self.batch_size)
            
    def train(self):
        """
        train the DQN model (network) while periodically updating the target model
        """
        
        while True:
            while self.episode_t < self.max_t_per_episode:
                action = self.get_action(self.state, test=False)
                done = self.step(action)
                self.epsilon = max(self.epsilon - 0.0009,
                                   self.epsilon_min)

                if self.t >= self.replay_start_t and self.t % 4 == 0:
                    
                    # sample
                    batch = list(map(list, zip(*self.sample(self.batch_size))))
                    states = np.array(batch[0])
                    actions = batch[1]
                    rewards = batch[2]
                    next_states = np.array(batch[3])
                    dones = tf.convert_to_tensor(batch[4])
                    
                    
                    next_q_vals = self.target_model.predict(next_states)
                    ys = rewards + self.gamma * tf.reduce_max(next_q_vals, axis=1)
                    ys = ys * (1 - dones) - dones
                    
                    masks = tf.one_hot(actions, self.act_size)
                    with tf.GradientTape() as tape:
                        q_vals = tf.reduce_sum(self.model(states) * masks, 
                                               axis=1)
                        loss = losses.Huber()(ys, q_vals)
                    
                    grads = tape.gradient(loss, 
                                          self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads,
                                                   self.model.trainable_variables))
                
                    print("loss: {}, timesteps: {}".format(loss, self.t))
                if self.t % self.target_update_period == 0:
                    self.target_model.set_weights(self.model.get_weights())

                
                if self.t % 5000 == 0:
                    self.save_model()
                
                self.t += 1
                if done:
                    break
                else:
                    self.episode_t += 1
            self.reset_episode()

                
                
                                
                