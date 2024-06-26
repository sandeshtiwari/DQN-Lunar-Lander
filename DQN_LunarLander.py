import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque

import random

np.random.seed(1)
tf.random.set_seed(1)



class Agent:
	def __init__(self, env, gamma=0.99, epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.9, lr=0.001, replay_memory_size=2000):
		self.env = env
		self.action_size = env.action_space.n
		self.state_size = env.observation_space.shape[0]
		self.gamma = gamma
		self.epsilon = epsilon
		self.final_epsilon = final_epsilon
		self.epsilon_decay = epsilon_decay
		self.lr = lr
		self.replay_memory_size = replay_memory_size
		self.model = self.create_nn_model()
		self.second_model = self.create_nn_model()
		self.replay_memory = deque(maxlen=replay_memory_size)
		self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done'))
		self.update_target()
		self.fill_replay()


	def update_epsilon(self):
		if self.epsilon > self.final_epsilon:
			self.epsilon *= self.epsilon_decay

	
	def create_nn_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(units=32, activation='relu'))
		model.add(tf.keras.layers.Dense(units=32, activation='relu'))
		# model.add(tf.keras.layers.Dense(units=16, activation='relu'))
		model.add(tf.keras.layers.Dense(units=self.action_size, activation='softmax'))
		model.build(input_shape=(None, self.state_size))
		model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
		return model

	def update_target(self):
		self.second_model.set_weights(self.model.get_weights())

	def store_replay(self, new_transition):
		self.replay_memory.append(new_transition)

	def select_action(self, state):
		# state = np.expand_dims(state, axis=0)
		if np.random.random() <= self.epsilon:
			return np.random.randint(self.action_size)
		actions = self.model.predict(state)[0]
		return np.argmax(actions)

	def fill_replay(self):
		state = self.env.reset()
		state = np.expand_dims(state, axis=0)
		for i in range(self.replay_memory_size-1500):
			action = self.select_action(state)
			next_state, reward, done, _ = self.env.step(action)
			next_state = np.expand_dims(next_state, axis=0)
			self.store_replay(self.transition(state, action, reward, next_state, done))
			if done:
				state = env.reset()
				state = np.expand_dims(state, axis=0)
			else:
				state = next_state

	def sample_replay(self, batch_size=32):
		batch_samples = random.sample(self.replay_memory, batch_size)
		return batch_samples

	def learn(self):
		batch_samples = self.sample_replay()
		batch_states, batch_targets = [], []
		for transition in batch_samples:
			s, a, r, next_s, done = transition
			s_x, s_y = s[0][0], s[0][1]
			s_dist = np.sqrt((s_x-0)**2 + (s_y-0)**2)

			next_s_x, next_s_y = next_s[0][0], next_s[0][1]
			next_s_dist = np.sqrt((next_s_x - 0)**2 + (next_s_y -0)**2)
			if done:
				target = r - s_dist + 0.99 * next_s_dist
			else:
				# print("S",s)
				target = (r - s_dist + 0.99 * next_s_dist + self.gamma * np.amax(self.second_model.predict(next_s)[0]))

				# target = (r + self.gamma * np.amax(self.second_model.predict(next_s)[0]))
			target_all = self.model.predict(s)[0]
			# print(next_s)
			# print("1 target_all", target_all, "/////////// state", s)
			target_all[a] = target
			# print("2 target_all", target_all, target)
			batch_states.append(s.flatten())
			batch_targets.append(target_all)
		# print("LIST TYPE", batch_states, "----", batch_targets)
		# print("FITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
		# print(np.array(batch_states), "-------", np.array(batch_targets))
		# print("**********************************************")
		history = self.model.fit(x=np.array(batch_states), y=np.array(batch_targets), verbose=1)
		self.update_epsilon()
		return history.history['loss'][0]

	def save_model(self):
		self.model.save('Model')


env = gym.make("LunarLander-v2")
env.mode = 'fast'
agent = Agent(env)
state = env.reset()
# print("Un Shaped", state)

# print("STATE", state)
# print(agent.model.summary())
# print(agent.second_model.summary())
# print(agent.select_action(state))
episodes = 50
losses = []
rewards = []
for epi in range(episodes):
	e_reward = 0
	print("In episode: ", epi)
	state = env.reset()

	state = np.expand_dims(state, axis=0)
	done = False
	# env.render()
	counter = 0
	while not done and counter < 500:
		# print("")
		counter += 1
		action = agent.select_action(state)
		next_state, reward, done, _ = env.step(action)
		e_reward += reward

		next_state = np.expand_dims(next_state, axis=0)
		agent.store_replay(agent.transition(state, action, reward, next_state, done))
		state = next_state
		# env.render()

		if done:
			rewards.append(e_reward)
			print("Episode ended with reward:"+str(e_reward))
			# agent.save_model()
		loss = agent.learn()
		losses.append(loss)
		if counter % 10 == 0:
			print("REWARDS", rewards)
	agent.update_target()
	# env.close()
print("REWARDS", rewards)
print("LOSSES", losses)
agent.save_model()