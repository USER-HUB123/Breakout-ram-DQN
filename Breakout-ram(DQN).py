import os, random, collections
import cv2, numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt

class DQN(tf.keras.Model):
    def __init__(self, input_shape, n_actions, max_memory_length=5000):
        super(DQN, self).__init__()
        self.network_layers = [
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_actions)
        ]

        self(tf.keras.Input(input_shape))
        self.summary()

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # create memory buffers
        self.max_memory_length = max_memory_length
        self.Transition = collections.namedtuple("Transition", ["state", "next_state", "action", "reward", "done"])
        self.memory = collections.deque([], maxlen=max_memory_length)
    
    def call(self, x):
        for layer in self.network_layers:
            x = layer(x)
        return x
    
    def memorize(self, state, next_state, action, reward, done):
        transition = self.Transition(state, next_state, action, reward, done)
        self.memory.append(transition)
    
    def sample_memory(self, batch_size):
        sample_batch = random.sample(self.memory, batch_size)
        sample_batch = self.Transition(*zip(*sample_batch))

        return (
            tf.convert_to_tensor(sample_batch.state, dtype=tf.float32),
            tf.convert_to_tensor(sample_batch.next_state, dtype=tf.float32),
            tf.convert_to_tensor(sample_batch.action, dtype=tf.int32),
            tf.convert_to_tensor(sample_batch.reward, dtype=tf.float32),
            tf.convert_to_tensor(sample_batch.done, dtype=tf.float32)
        )

class Env:
    def __init__(self, max_memory_length=1000000, min_memory=32, eps_0=1.0, eps_min=.05, eps_decay=.99999, initial_frames=0):
        """
        eps: (epsilon) chance of taking a random action
        eps_random_frame: number of steps during the initial training phase where the epsilon is decayed
        eps_interval: minimum epsilon value
        """
        self.max_memory_length = max_memory_length
        self.min_memory = min_memory
        self.eps_0 = self.eps = eps_0
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # create environment
        self.frames_count = initial_frames
        self.env = gym.make("ALE/Breakout-ram-v5")
        self.model = DQN(self.env.observation_space.shape, self.env.action_space.n, max_memory_length=self.max_memory_length)
        self.target_model = DQN(self.env.observation_space.shape, self.env.action_space.n)

    def process_frame(self, frame):
        return tf.convert_to_tensor(frame.astype(np.float32) / 255)

    def train(self, training_frequency=4, target_network_update_frequency=10000, batch_size=32, max_steps=10000, save_steps=1000):
        """
        training_frequency: number of frames after which the training process is executed
        target_network_update_frequency: number of frames after which the target network is updated
        """
        self.env = gym.make("ALE/Breakout-ram-v5")
        self.target_model.set_weights(self.model.get_weights())
        episode, best_reward, rewards = 0, 0, collections.deque([])
        while True:
            episode += 1
            done = False
            episode_reward = 0

            # initialize environment
            state = self.process_frame(self.env.reset()[0])

            for fr in range(max_steps):
                if done: break
                self.frames_count += 1

                # select action and decay epsilon value
                if random.random() < self.eps: action = np.random.choice(self.env.action_space.n)
                else: action = np.argmax(np.squeeze(self.model(state[None, ...], training=False), axis=0))
                if self.frames_count % 50 == 0:
                    self.eps = max(self.eps_min, self.eps_0 * self.eps_decay**(self.frames_count/6))

                # apply actions and memorize transition data
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.process_frame(next_state)
                episode_reward += reward
                self.model.memorize(state, next_state, action, reward, done)
                state = next_state

                # training the model (only after certain conditions are met)
                if self.frames_count % training_frequency == 0 and len(self.model.memory) > self.min_memory:
                    samples = self.model.sample_memory(batch_size)
                    self.train_step(samples)

                # equalize target model and training model
                if self.frames_count % target_network_update_frequency == 0:
                    self.target_model.set_weights(self.model.get_weights())
                
                #save training history data
                if self.frames_count % save_steps == 0:
                    fig, ax = plt.subplots()
                    ax.plot(*list(zip(*rewards)))
                    fig.savefig("Breakout-ram(DQN) Rewards.jpg")
                    plt.close()
            
            if episode_reward >= best_reward:
                best_reward = episode_reward
                self.model.save_weights(f"Breakout-ram(DQN).h5")

            # report training progress
            rewards.append((self.frames_count, episode_reward))
            print(f"\r Episode {episode}; steps: {self.frames_count}; Running reward: {episode_reward:.2f}.", end="")

    #@tf.function
    def train_step(self, samples, discount_factor=.99):
        #sample training data from collected memory
        states_sample, next_states_sample, actions_sample, rewards_sample, done_sample = samples

        # build the updated Q-values for the sampled future states using the target model
        future_rewards = self.target_model(next_states_sample)
        future_rewards = tf.math.reduce_max(future_rewards, axis=1)
        updated_q_values = rewards_sample + discount_factor * future_rewards * (1 - done_sample)
        mask = tf.one_hot(actions_sample, depth=self.env.action_space.n)

        with tf.GradientTape() as tape:
            
            # compute the network's loss
            q_values = self.model(states_sample); print(q_values, tf.argmax(q_values, axis=1)); exit()
            q_actions = tf.reduce_sum((q_values * mask), axis=1)
            loss = self.model.loss_fn(updated_q_values, q_actions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def demonstrate(self, eps=0):
        self.env = gym.make("ALE/Breakout-ram-v5", render_mode="human")
        while True:
            done = False
            episode_reward = 0
            state = self.process_frame(self.env.reset()[0])

            while not done:
                self.env.render()
                if random.random() < eps: action = np.random.choice(self.env.action_space.n)
                else: action = np.argmax(np.squeeze(self.model(state[None, ...], training=False), axis=0))

                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                state = self.process_frame(state)
                episode_reward += reward

            print(f"\r Episode reward: {episode_reward:.2f}.", end="")

if __name__ == "__main__":
    env = Env()
    if os.path.exists("Breakout-ram(DQN).h5"): env.model.load_weights("Breakout-ram(DQN).h5")

    rewards_hist = env.train(save_steps=1000) # train the model ad infinitum
    env.demonstrate() # demonstrate the model in a test run