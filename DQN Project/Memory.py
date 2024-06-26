import numpy as np
import torch

class ExperienceBuffer:
	def __init__(self, input_dims, memory_size=100000):
		self.memory_size = memory_size
		self.state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.memory_size, *input_dims), dtype=np.float32)

		self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

		self.memory_counter = 0
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = torch.device(device)

	def append(self, state, action, reward, next_state, done):
		index = self.memory_counter % self.memory_size
		self.state_memory[index] = state
		self.new_state_memory[index] = next_state
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = done

		self.memory_counter += 1

	def sample(self, batch_size=64):
		max_mem = min(self.memory_counter, self.memory_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)

		batch_index = np.arange(batch_size, dtype=np.int32)

		state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
		new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)
		reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
		terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)
		action_batch = self.action_memory[batch]
		return state_batch, new_state_batch, reward_batch, terminal_batch, action_batch, batch_index