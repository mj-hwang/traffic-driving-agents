import numpy as np
import tensorflow as tf
from scipy.linalg import solve_continuous_are as sol_riccatti
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

tf.keras.backend.set_floatx('float64')

class SystemModel(Model):
    # Approximate the LTV system 
    def __init__(self, obs_size, act_size):
        super(SystemModel, self).__init__()
        self.A1 = Dense(100, activation='relu')
        self.A2 = Dense(100, activation='relu')
        self.A3 = Dense(obs_size * obs_size)
        
        self.B1 = Dense(100, activation='relu')
        self.B2 = Dense(100, activation='relu')
        self.B3 = Dense(obs_size * act_size)
        
        self.obs_size = obs_size
        self.act_size = act_size

    def call(self, xu):
        x, u = np.hsplit(xu, np.array([self.obs_size]))
        A, B = self.get_system(xu) # shape: (batch_size, obs_size, obs_size)
        dx = tf.matmul(A, tf.expand_dims(x, -1)) + \
             tf.matmul(B, tf.expand_dims(u, -1))
        dx = tf.squeeze(dx, [2])
        return x+dx
        
    def get_system(self, xu):
        A = tf.reshape(self.A3(self.A2(self.A1(xu))), 
                       [-1, self.obs_size, self.obs_size])
        B = tf.reshape(self.B3(self.B2(self.B1(xu))), 
                       [-1, self.obs_size, self.act_size])
        return A, B
        
class LQRAgent():
    def __init__(self, env):
        self.env = env
        self.obs_size = env.observation_space['observation'].shape[0]
        self.act_size = env.action_space.shape[0]
        self.system_model = SystemModel(self.obs_size, self.act_size)
        self.weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        self.train_weights = tf.convert_to_tensor(self.weights) ** 0.5
    
    def train(self, buffer_size=50, num_epochs=100):
        buffer = []
        while len(buffer) < buffer_size:
            x = self.env.reset()
            done = False
            while not done:
                u = self.env.action_space.sample()
                x_next, _, _, done = self.env.step(u)
                buffer.append((x['observation'], u, x_next['observation']))
                x = x_next
        
        data = list(map(np.array, zip(*buffer)))
        x_lst = data[0] 
        u_lst = data[1] 
        x_next_lst = data[2] 
        xu_lst = np.hstack((x_lst, u_lst))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
        
        loss_fn = tf.keras.losses.MeanSquaredError()                   
        trainset = tf.data.Dataset.from_tensor_slices((xu_lst, x_next_lst))
        trainset = trainset.shuffle(buffer_size).batch(32)

        # Iterate over epochs.
        for epoch in range(num_epochs):
            print("Start of epoch %d" % (epoch,))
            
            for step, (xu_batch, x_next_batch) in enumerate(trainset):
                with tf.GradientTape() as tape:
                    loss = loss_fn(self.system_model(xu_batch)*self.train_weights, 
                                   x_next_batch*self.train_weights)
                    
                grads = tape.gradient(loss, self.system_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.system_model.trainable_weights))
                print(loss.numpy())
               
        
    def get_system(self, x, u):
        x_copy = x.copy()
        A, B = self.system_model.get_system(tf.expand_dims(np.hstack((x, u)),
                                                           axis=0))
        
        A = tf.squeeze(A).numpy()
        B = tf.squeeze(B).numpy()
                
        # add identity matrix to A
        A = np.eye(self.obs_size) + A
        
        return A, B

    def get_action(self, state, goal=None, Q=None, R=None):
        if goal == None:
            goal = np.zeros(self.sim_env.state.shape)
        u = np.zeros(self.sim_env.action_space.shape)
        x = state['observation']
#         A = self.approximate_A(x, u)
#         B = self.approximate_B(x, u)
        A, B = self.get_system(self, x, u)
        Q = np.diag(self.weigths)
        R = -0.3 * np.eye(self.act_size)
        P = sol_riccatti(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        u = -K @ (x-np.copy(goal))
        return u