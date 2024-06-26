import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

import os

import gym

from Memory import ExperienceBuffer

class DQN(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
		super(DQN, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = torch.device(device)
		self.to(self.device)


	def forward(self, state):
		# print("STATE", state)
		# print("STATE SHAPE", state.shape)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		actions = self.fc3(x)
		# print(actions.shape)

		return actions


class Agent():
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, eps_end=0.01, max_mem_size=100000, eps_dec=0.00005, update_timestamp=100):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_min = eps_end
		self.eps_dec = eps_dec
		self.lr = lr
		self.action_space = [i for i in range(n_actions)]
		self.mem_size = max_mem_size
		self.batch_size = batch_size
		self.memory_counter = 0
		self.update_timestamp = update_timestamp

		self.Q_network = DQN(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
		self.target_Q_network = DQN(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

		self.update_target()

		self.experience_buffer = ExperienceBuffer(input_dims=input_dims)

		self.c = 0

	def update_target(self):
		self.target_Q_network.load_state_dict(self.Q_network.state_dict())

	def store_transition(self, state, action, reward, next_state, done):
		self.experience_buffer.append(state, action, reward, next_state, done)

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = torch.tensor([observation]).to(self.Q_network.device)
			actions = self.Q_network.forward(state)
			action = torch.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def learn(self):
		if self.experience_buffer.memory_counter < self.batch_size:
			return
		self.c += 1
		# print("Learning...")

		self.Q_network.optimizer.zero_grad()

		state_batch, new_state_batch, reward_batch, terminal_batch, action_batch, batch_index = self.experience_buffer.sample(self.batch_size)


		# s_x, s_y = state_batch[0][0], state_batch[0][1]
		# s_dist = np.sqrt((s_x-0)**2 + (s_y-0)**2)

		# next_s_x, next_s_y = new_state_batch[0][0], new_state_batch[0][1]
		# next_s_dist = np.sqrt((next_s_x - 0)**2 + (next_s_y -0)**2)



		Q_network = self.Q_network.forward(state_batch)[batch_index, action_batch]
		# target network for q_next
		q_next = self.target_Q_network.forward(new_state_batch)
		# print("q_next",q_next[terminal_batch])
		q_next[terminal_batch] = 0.0

		q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
		# q_target = reward_batch - s_dist + 0.99 * next_s_dist + self.gamma * torch.max(q_next, dim=1)[0]

		loss = self.Q_network.loss(q_target, Q_network).to(self.Q_network.device)
		loss.backward()
		self.Q_network.optimizer.step()

		if self.epsilon > self.eps_min:
			self.epsilon = self.epsilon - self.eps_dec	
		else:
			self.epsilon = self.eps_min

		if self.c == self.update_timestamp:
			self.update_target()
			self.c = 0

	def write_values_to_file(self, file_):
		pass

	def save_model(self, name):
		FILE = 'model_'+name+'.pt'
		torch.save(self.Q_network.state_dict(), FILE)

	# def load_model(self):
		# model = DQN()
		# model.load_state_dict(torch.load(PATH))
		# model.eval()
