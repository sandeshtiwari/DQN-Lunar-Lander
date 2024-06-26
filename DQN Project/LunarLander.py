import gym
from DQN import Agent
import numpy as np
import matplotlib.pyplot as plt
import time
from DQN import DQN
import torch

def graph(episodes_values, color, facecolor, all_average, all_rewards, name, fig_name):
	plt.figure(figsize=(9,7))
	plt.title('DQN',size=15)
	plt.ylabel('Rewards',size=20)
	plt.xlabel('Episodes',size=20)
	all_average = np.array(all_average)
	all_rewards = np.array(all_rewards)
	plt.plot(episodes_values,all_average, label=name, color=color)
	# plt.fill_between(episodes_values, all_average-all_rewards, all_average+all_rewards, edgecolor=color, facecolor=facecolor, alpha=0.5)
	plt.legend(loc='upper left')
	plt.savefig(fig_name)
	plt.close()

def run_trained_agent(PATH):
	# PATH = "model.pt"
	model = DQN(n_actions=4, input_dims=[8], fc1_dims=256, fc2_dims=256, lr=0.001)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	model.load_state_dict(torch.load(PATH))
	model.eval()


	env = gym.make('LunarLander-v2')
	# print(env.action_space.sample())

	EPISODES = 100
	all_rewards = []
	all_average = []
	for i in range(EPISODES):
		state = env.reset()
		epi_reward = 0
		env.render()
		done = False
		while not done:
			state = torch.tensor(state).to(device)
			actions = model.forward(state.view(1,8))
			# print("prediction",actions)
			# break
			action = torch.argmax(actions[0])
			# print("ACTION INDEX", action, action.item())
			# break
			next_state, reward, done, _ = env.step(action.item())
			epi_reward += reward
			state = next_state
			env.render()
			# break
		print('episode', i, 'reward % .2f' % epi_reward)
		
		# break
		all_rewards.append(epi_reward)
		avg_score = np.mean(all_rewards[-100:])
		all_average.append(avg_score)
		print("Average last 100 reward", avg_score)

	print("Average reward for the trained agent",np.mean(all_rewards))
	episodes_values = [i for i in range(EPISODES)]
	graph(episodes_values, '#1B2ACC', '#089FFF', all_average, all_rewards, name="Final Model", fig_name="Trained_Model.png")

	# file1 = open("Trained_Model.txt", "a")  # append mode 
	# file1.write("all_average ="+str(all_average)+"\n")
	# file1.write("all_rewards ="+str(all_rewards)) 
	# file1.close()
	env.close()

			

def train_new_agent(plt, file_,color, facecolor, fig_name, name, gamma=0.99, lr=0.001, eps_dec=0.00005):
	env = gym.make('LunarLander-v2')
	agent = Agent(gamma=gamma, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=lr, eps_dec=eps_dec)
	print("Using device:", agent.Q_network.device)
	all_rewards, eps_history = [], []
	all_average = []
	EPISODES = 500

	# start_time = time.time()
	checked = False
	for i in range(EPISODES):
		epi_reward = 0
		done = False
		state = env.reset()
		env.render()
		while not done:
			# print("EPI")
			action = agent.choose_action(state)
			next_state, reward, done, _ = env.step(action)
			env.render()
			epi_reward += reward
			# print("HERE")
			agent.store_transition(state, action, reward, next_state, done)
			# print("HERE....")
			agent.learn()
			state = next_state
		all_rewards.append(epi_reward)
		eps_history.append(agent.epsilon)
		# if i == 2:
		# 	break

		avg_score = np.mean(all_rewards[-100:])
		all_average.append(avg_score)
		# if avg_score > 200 and not checked:
		# 	end_time = time.time()
		# 	checked = True

		print('episode', i, 'reward % .2f' % epi_reward, 'average reward %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon, '-- For %s' % name)
	# end_time = time.time()
	# time_min = (end_time - start_time)/60
	# print("--- %s minutes ---" % time_min)
	agent.save_model(name)
	# print([i for i in range(3)], all_rewards)
	# plt.figure(figsize=(9,7))
	all_average = np.array(all_average)
	all_rewards = np.array(all_rewards)
	episodes_values = [i for i in range(EPISODES)]
	
	
	# plt.figure(figsize=(9,7))
	# plt.title('DQN',size=15)
	# plt.ylabel('Rewards',size=20)
	# plt.xlabel('Episodes',size=20)
	# plt.plot(episodes_values,all_average, label=name, color=color)
	# plt.fill_between(episodes_values, all_average-(all_rewards), all_average+(all_rewards), edgecolor=color, facecolor=facecolor, alpha=0.5)
	# plt.legend(loc='upper left')
	# plt.savefig(fig_name)
	# plt.close()

	file1 = open(file_, "a")  # append mode 
	file1.write("all_average ="+str(all_average)+"\n")
	file1.write("all_rewards ="+str(all_rewards)) 
	file1.close() 
	

	env.close()

if __name__ == '__main__':
	option = int(input("Enter 1 to train a new model and 2 to use existing agent: "))
	# print(option)
	if option == 1:
		colors = ['#1B2ACC', '#CC4F1B', '#3F7F4C']
		facecolors = ['#089FFF','#FF9848', '#7EFF99']

		# Gamma experiment
		# plt.figure(figsize=(9,7))
		# plt.title('DQN γ',size=15)
		# plt.ylabel('Rewards',size=20)
		# plt.xlabel('Episodes',size=20)
		gammas = [0.95, 0.97, 0.99]
		for i, gamma in enumerate(gammas):
			print("Training for GAMMA: ", gamma)
			name = "GAMMA_"+str(gamma)
			train_new_agent(file_=name+".txt", plt=plt, gamma=gamma, color=colors[i], facecolor=facecolors[i], fig_name=name+".png", name=name)
		# plt.legend(loc='upper left')
		# plt.savefig('Gamma_experiment.png')
		# plt.close()

		# Learning rates experiment
		# plt.figure(figsize=(9,7))
		# plt.title('DQN α',size=15)
		# plt.ylabel('Rewards',size=20)
		# plt.xlabel('Episodes',size=20)


		lrs = [0.0009, 0.001, 0.002]
		for i, lr in enumerate(lrs):
			print("Training for lr: ", lr)
			name = "ALPHA_"+str(lr)
			train_new_agent(file_=name+".txt", plt=plt, lr=lr, color=colors[i], facecolor=facecolors[i],fig_name=name+".png", name=name)
		# print("Training for lr: ", 0.001)
		# name = "ALPHA_"+str(0.001)
		# train_new_agent(file_=name+".txt", plt=plt, lr=0.001, color=colors[1], facecolor=facecolors[1],fig_name=name+".png", name=name)

		# plt.legend(loc='upper left')
		# plt.savefig('Learning_Rate_experiment.png')
		# plt.close()


		# # Episilon decay experiment
		# plt.figure(figsize=(9,7))
		# plt.title('DQN ε',size=15)
		# plt.ylabel('Rewards',size=20)
		# plt.xlabel('Episodes',size=20)
		epsilons_decays = [0.000005, 0.00005, 0.0005]
		for i, epsilon_decay in enumerate(epsilons_decays):
			print("Training for epsilon decay: ", epsilon_decay)
			name = "EPSILON_DECAY"+str(epsilon_decay)
			train_new_agent(file_=name+".txt", plt=plt, eps_dec=epsilon_decay, color=colors[i], facecolor=facecolors[i],fig_name=name+".png", name=name)
		# plt.legend(loc='upper left')
		# plt.savefig('Epsilon_experiment.png')
		# plt.close()
	elif option == 2:
		run_trained_agent("ALPHA_0.001.pt")
