import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import random 
from collections import deque
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
        # self.rnn.flatten_parameters() # 确保权重在内存中连续
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
        self.GRUmodel = Transformer(hidden_size = 1 ) # GRUModel(hidden_size = self.hidden_size ) # GRUModel Transformer
        # Transformer(hidden_size = self.hidden_size+1 ) # GRUModel(hidden_size = self.hidden_size ) # 
        
        # self.l1 = nn.Linear(self.hidden_size+1, 256)  # state_dim
        # self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(self.hidden_size+1, action_dim) # self.hidden_size+1
        # self.max_action = max_action
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state):
        X = self.GRUmodel(state)
        # a = F.relu(X) #F.relu(self.l1(state))
        # x = self.l3(a)
        # x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        x = self.sigmoid(X)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hidden_size = 128 
        self.GRUmodel = Transformer(hidden_size = 1 ) # GRUModel(hidden_size = self.hidden_size ) # GRUModel Transformer
        # Transformer(hidden_size = self.hidden_size+1 ) # GRUModel(hidden_size = self.hidden_size ) # GRUModel Transformer
        # Q1 architecture
        # self.l1 = nn.Linear(self.hidden_size+1, 256)  # state_dim + action_dim
        # self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(self.hidden_size+1, 1) # self.hidden_size+1

    def forward(self, state):
        Q = self.GRUmodel(state)
        # q1 = F.relu(Q) # F.relu(self.l1(state))
        # q1 = self.l3(q1)
        return Q #, q2

'''
    def Q1(self, state, action):
        # self.GRUmodel.flatten_parameters()  # 确保权重在内存中连续
        state = self.GRUmodel(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
'''

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
    def __init__(self, hidden_size=128, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.1, device=None):
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
        # self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2e-4)
        # itertools.chain(self.GRUmodel.parameters(),
        self.critic = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)
                
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
        self.start_timesteps = 2 #5e3
        self.expl_noise = 0.1
        self.batch_size = 1280 #256
        self.device = device
        self.alpha = 2.5

    def get_action(self, state):
        state = state.reshape(1, -1).to(dtype=torch.float32, device=device)
        # state = self.GRUmodel(state)        
        return self.actor(state).cpu().data.numpy() #.flatten()

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

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha/Q.abs().mean().detach()
            # actor_loss = - Q.mean() 
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
            
            # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            #  + F.mse_loss(pi, action) 
   
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

class PPO:
    def __init__(self, state_dim,
        action_dim=2,
        max_action=1,
        device= 'cuda'):
        
        self.gamma = 0.99  # 折扣因子
        self.actor_lr = 2e-4 #1e-3  # 策略网络的学习率 2e-4 #
        self.critic_lr = 5e-4 #1e-2  # 价值网络的学习率 5e-4 #
        self.batch_size = 128 # 1000 #256

        # 实例化策略网络
        # self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        self.actor = Actor(state_dim = state_dim, action_dim=action_dim, max_action=max_action).to(device)
        
        # 实例化价值网络
        # self.critic = ValueNet(n_states, n_hiddens).to(device)
        self.critic = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        # 定义学习率调度器
        self.Re_scheduler_a = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.Re_scheduler_c = ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        self.lmbda = 0.95  # GAE优势函数的缩放系数
        self.epochs = 1 #5  # 一条序列的数据用来训练轮数
        self.eps = 0.2  # PPO中截断范围的参数
        self.device = device

    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = state.reshape(1, -1).to(dtype=torch.float32, device=device)
        # state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state).item()
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(torch.tensor([1-probs,probs]))
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action, probs

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        def batch_sample(data_list, batch_size):
            # 打乱顺序
            random.shuffle(data_list)
            # 将数据按照 batch size 切分
            for i in range(0, len(data_list), batch_size):
                yield data_list[i:i + batch_size]

        # 迭代抽样
        for mini_batch in batch_sample(transition_dict, self.batch_size):
            states, actions, rewards, next_states, dones = zip(*mini_batch)

            # 将它们转换为 NumPy 数组
            states = np.vstack(states)
            actions = np.vstack(actions)
            rewards = np.vstack(rewards)
            next_states = np.vstack(next_states)
            dones = np.vstack(dones).astype(int)  # bool to binary

            # 将 NumPy 数组转换为 PyTorch 张量，并移动到 CUDA 上
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            # 目标，下一个状态的state_value  [b,1]
            next_q_target = self.critic(next_states)
            # 目标，当前状态的state_value  [b,1]
            td_target = rewards + self.gamma * next_q_target * (1-dones)
            # 预测，当前状态的state_value  [b,1]
            td_value = self.critic(states)
            # 目标值和预测值state_value之差  [b,1]
            td_delta = td_target - td_value

            # 时序差分值 tensor-->numpy  [b,1]
            td_delta = td_delta.cpu().detach().numpy()
            advantage = 0  # 优势函数初始化
            advantage_list = []

            # 计算优势函数
            for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
                # 优势函数GAE的公式
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append(advantage)
            # 正序
            advantage_list.reverse()
            # numpy --> tensor [b,1]
            advantage = torch.tensor(np.array(advantage_list), dtype=torch.float32).to(self.device)

            # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
            prob = self.actor(states)
            old_log_probs = torch.log(prob*actions+(1-prob)*(1-actions)).detach()
            # old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            # 一组数据训练 epochs 轮
            for _ in range(self.epochs):
                # 每一轮更新一次策略网络预测的状态
                prob = self.actor(states)
                log_probs = torch.log(prob*actions+(1-prob)*(1-actions))
                # log_probs = torch.log(self.actor(states).gather(1, actions))
                # 新旧策略之间的比例
                ratio = torch.exp(log_probs - old_log_probs)
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage
                # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

                # 策略网络的损失函数
                actor_loss = torch.mean(-torch.min(surr1, surr2))
                # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

                # 梯度清0
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # 反向传播
                actor_loss.backward()
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), 3.)
                critic_loss.backward()
                torch.nn.utils.clip_grad_value_(self.critic.parameters(), 3.)
                # 梯度更新
                self.actor_optimizer.step()
                self.critic_optimizer.step()
