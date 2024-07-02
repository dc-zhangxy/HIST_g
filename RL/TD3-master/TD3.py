import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action=1,
		discount=0.99,
		tau=0.005,
		device = 'cuda',
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

		# create replay memory using deque
		self.memory_size = int(1e6)
		self.memory = deque(maxlen=self.memory_size)
		self.start_timesteps = 25e3
		self.expl_noise = 0.1
		self.batch_size = 256
		self.device = device


	def get_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data #.numpy().flatten()

	# def get_action(self, state, t):
	# 	# Select action randomly or according to policy
	# 	if t < self.start_timesteps:
	# 		action = random.uniform(-self.max_action, self.max_action) #env.action_space.sample()
	# 	else:
	# 		action = (
	# 			self.select_action(np.array(state))
	# 			+ np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
	# 		).clip(-self.max_action, self.max_action)
	# 	return action

	def train_model(self, ):
		self.total_it += 1

		# Sample replay buffer 
		# state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		mini_batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*mini_batch)

		# 将它们转换为 NumPy 数组
		states = np.vstack(states)
		actions = np.vstack(actions)
		rewards = np.vstack(rewards)
		next_states = np.vstack(next_states)
		dones = np.vstack(dones).astype(int)  # bool to binary

		# 将 NumPy 数组转换为 PyTorch 张量，并移动到 CUDA 上
		state = torch.tensor(states, dtype=torch.float32).to(self.device)
		action = torch.tensor(actions, dtype=torch.float32).to(self.device)
		reward = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		next_state = torch.tensor(next_states, dtype=torch.float32).to(self.device)
		done = torch.tensor(dones, dtype=torch.float32).to(self.device)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (1 - done) * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
		# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		# 将批次数据逐个拆开并存入经验回放池
		# for i in range(state.shape[0]): # batch size
		#     self.memory.append((state[i], action[i], reward[i], next_state[i], done[i]))
		self.memory.append((state, action, reward, next_state, done))
