import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from collections import deque
import collections
import math
import itertools 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class GRUModel(nn.Module):
    def __init__(self, hidden_size=128, d_feat=6, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # self.fc_out = nn.Linear(hidden_size+1, action_size)

        self.d_feat = d_feat

    def forward(self, x):
        self.rnn.flatten_parameters() # 确保权重在内存中连续
        # feature: [N, F*T]
        feature = x[:,:-1]
        xt = x[:,-1:]
        feature = feature.reshape(len(feature), self.d_feat, -1)  # [N, F, T]
        feature = feature.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(feature)
        ret = torch.concat([out[:, -1, :], xt],1)
        # ret = self.fc_out(torch.concat([out[:, -1, :], xt],1)) #.squeeze() 会把[1,5]的维度变成[5]
        return ret

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_size = 128 
        self.GRUmodel = GRUModel(hidden_size = self.hidden_size )  # Transformer(hidden_size = self.hidden_size+1 ) #  GRUModel(hidden_size = self.hidden_size ) # Transformer(action_size = action_size) #  DQN(state_size, action_size) #
        # GRUModel(hidden_size = self.hidden_size ) 
        self.l1 = nn.Linear(self.hidden_size+1, 256)  # state_dim
        self.l2 = nn.Linear(self.hidden_size+1, action_dim)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        state = self.GRUmodel(state)
        a = F.relu(self.l1(state))
        ret = self.max_action * torch.tanh(self.l3(a))
        # a = self.l2(F.relu(state))
        # ret = self.max_action * torch.tanh(a)
        return ret


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hidden_size = 128 
        self.GRUmodel = GRUModel(hidden_size = self.hidden_size )  # Transformer(hidden_size = self.hidden_size+1 ) # GRUModel(hidden_size = self.hidden_size )
        # GRUModel(hidden_size = self.hidden_size ) #
        # Q1 architecture
        self.l1 = nn.Linear(self.hidden_size+1 + action_dim, 256)  # state_dim + action_dim
        self.l2 = nn.Linear(self.hidden_size+1 + action_dim, 1)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(self.hidden_size+1 + action_dim, 256)  # state_dim + action_dim
        self.l5 = nn.Linear(self.hidden_size+1 + action_dim, 1)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        state = self.GRUmodel(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = self.l6(q2)
        # q1 = self.l2(F.relu(sa))
        # q2 = self.l5(F.relu(sa))
        return q1, q2


    def Q1(self, state, action):
        # self.GRUmodel.flatten_parameters()  # 确保权重在内存中连续
        state = self.GRUmodel(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l3(q1)
        # q1 = self.l2(F.relu(sa))
        return q1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, hidden_size=2, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.1, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model+1, hidden_size)
        self.device = device
        self.d_feat = d_feat
        # self.fc_out = nn.Linear(hidden_size+1, action_size)

    def forward(self, src_input):
        feature = src_input[:,:-1]
        xt = src_input[:,-1:]
        # feature [N, F*T] --> [N, T, F]
        feature = feature.reshape(len(feature), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(feature)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first
        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = output.transpose(1, 0)[:, -1, :]  # [512, 1]
        output = self.decoder_layer(torch.concat([output, xt],1))

        return output #.squeeze()


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
        self.hidden_size = 128 
        # self.GRUmodel = GRUModel(hidden_size = self.hidden_size ).to(device) # Transformer(action_size = action_size) #  DQN(state_size, action_size) #
        self.actor = Actor(state_dim = state_dim, action_dim=action_dim, max_action=max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2e-4)
        # itertools.chain(self.GRUmodel.parameters(),
        self.critic = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)

          # 定义学习率调度器
        # Re_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=25, verbose=True)
        # best_param = copy.deepcopy(model.state_dict())
        
        self.smooth_steps = 5
        self.actor_params_list = collections.deque(maxlen=self.smooth_steps) #5
        self.critic_params_list = collections.deque(maxlen=self.smooth_steps) #5
        self.avg = 45
          # pprint('evaluating...')
        # model.load_state_dict(params_ckpt)
        # 使用调度器调整学习率
        # Re_scheduler.step(-val_score)
           # best_param = copy.deepcopy(avg_params)
                
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        # create replay memory using deque
        self.memory_size = int(2e4)
        self.memory = deque(maxlen=self.memory_size)
        self.start_timesteps = 2 #5e3
        self.expl_noise = 0.1
        self.batch_size = 256 # 64 # 128 # 1280 #256
        self.device = device
        self.alpha = 2.5


    def get_action(self, state):
        state = state.reshape(1, -1).to(dtype=torch.float32, device=device)
        # state = self.GRUmodel(state)        
        return self.actor(state).cpu().data.numpy() #.flatten()

    def train_model(self, epoch):
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
  
        # state = self.GRUmodel(state)
        # next_state = self.GRUmodel(next_state)

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
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        if epoch>=self.avg:
            params_ckpt = copy.deepcopy(self.critic.state_dict())
            self.critic_params_list.append(params_ckpt)
            avg_params = average_params(self.critic_params_list)
            self.critic.load_state_dict(avg_params)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            actor_loss = - Q.mean() 
            # lmbda = self.alpha/Q.abs().mean().detach()
            # actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
            
            # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            #  + F.mse_loss(pi, action) 
   
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            if epoch >=self.avg:
                params_ckpt = copy.deepcopy(self.actor.state_dict())
                self.actor_params_list.append(params_ckpt)
                avg_params = average_params(self.actor_params_list)
                self.actor.load_state_dict(avg_params)
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

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params