import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from collections import deque
import math
import itertools 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.GRUmodel = GRUModel(hidden_size = self.hidden_size ) # Transformer(action_size = action_size) #  DQN(state_size, action_size) #
        #  Transformer(hidden_size = self.hidden_size+1 ) #
        self.l1 = nn.Linear(self.hidden_size+1, 256)  # state_dim
        self.l2 = nn.Linear(256, action_dim)
        self.l3 = nn.Linear(256, action_dim)
        # self.max_action = max_action
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        
    def forward(self, state, deterministic, with_logprob):
        state = self.GRUmodel(state)
        x = F.relu(self.l1(state))
        # a = F.relu(self.l2(a))
        mu =  torch.tanh(self.l2(x)) # self.l2(a) #
        log_std = torch.tanh(self.l3(x)) # self.l3(a) #
        # x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        if deterministic: 
            u = mu
            a = u 
        else: 
            u = dist.rsample()
            a = torch.tanh(u)
        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        # a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None
        return a, logp_pi_a 

    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super(Critic, self).__init__()
        self.hidden_size = 128 
        self.GRUmodel = GRUModel(hidden_size = self.hidden_size )
        # Transformer(hidden_size = self.hidden_size+1 ) 
        # Q1 architecture
        self.l1 = nn.Linear(self.hidden_size+1+ action_dim, 256)  # state_dim + action_dim
        # self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state, action):
        state = self.GRUmodel(state)
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1 #, q2


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

class SAC:
    def __init__(self, state_dim, action_dim, max_action=1, 
                 device = torch.device('cuda') ):
        # 属性分配
        self.gamma = 0.99
        self.tau = 0.005 # 软更新参数
        self.num_epochs = 1  # 训练回合数
        # self.capacity = 500  # 经验池容量
        # self.min_size = 200 # 经验池训练容量
        self.batch_size = 256 # 128 #64
        # self.n_hiddens = 64
        self.actor_lr = 2e-4 # 1e-3  # 策略网络学习率
        self.critic_lr = 5e-4 # 1e-2  # 价值网络学习率
        self.alpha_lr = 5e-4 #1e-2  # 课训练变量的学习率
        self.target_entropy = -1
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.device = device
        # 实例化策略网络
        # self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        self.actor = Actor(state_dim = state_dim, action_dim=action_dim, max_action=max_action).to(device)
        
        # 实例化第一个价值网络--预测
        # self.critic_1 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        self.critic_1 = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        # 实例化第二个价值网络--预测
        # self.critic_2 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        self.critic_2 = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        
        # 实例化价值网络1--目标
        # self.target_critic_1 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        self.target_critic_1 = Critic(state_dim = state_dim, action_dim=action_dim).to(device)
        # 实例化价值网络2--目标
        # self.target_critic_2 = ValueNet(n_states, n_hiddens, n_actions).to(device)
        self.target_critic_2 = Critic(state_dim = state_dim, action_dim=action_dim).to(device)

        # 预测和目标的价值网络的参数初始化一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # 目标网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float) # 0.2
        # alpha可以训练求梯度
        self.log_alpha.requires_grad = True
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    # 动作选择
    def take_action(self, state, deterministic, with_logprob):  # 输入当前状态 [n_states]
        # 维度变换 numpy[n_states]-->tensor[1,n_states]
        # state = torch.tensor(state[np.newaxis,:], dtype=torch.float).to(self.device)
        state = state.reshape(1, -1).to(dtype=torch.float32, device=self.device)
        # 预测当前状态下每个动作的概率  [1,n_actions]
        a, logp_pi_a = self.actor(state, deterministic, with_logprob)
        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
        # # we learn log_std rather than std, so that exp(log_std) is always > 0
        # std = torch.exp(log_std)
        # dist = torch.distributions.Normal(mu, std)
        # if deterministic: 
        #     u = mu
        #     # a = u 
        # else: 
        #     u = dist.rsample()
        #     # a = torch.tanh(u)
        # '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        # a = torch.tanh(u)
        # if with_logprob:
        #     # Get probability density of logp_pi_a from probability density of u:
        #     # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
        #     # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
        #     logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        # else:
        #     logp_pi_a = None

        return a.cpu().data.numpy(), logp_pi_a
        # 构造与输出动作概率相同的概率分布
        # action_dist = torch.distributions.Categorical(probs)
        # 从当前概率分布中随机采样tensor-->int
        # action = action_dist.sample().item()
        # return action
    
    # 计算目标，当前状态下的state_value
    def calc_target(self, rewards, next_states, dones):
        # 策略网络预测下一时刻的state_value  [b,n_states]-->[b,n_actions]
        a, log_pi_a  = self.actor(next_states, deterministic=False, with_logprob=True)
        # 对每个动作的概率计算ln  [b,n_actions]
        # next_log_probs = torch.log(next_probs + 1e-8)
        # 计算熵 [b,1]
        # entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdims=True)
        # 目标价值网络，下一时刻的state_value [b,n_actions]
        # print(next_states.shape,a.shape)
        q1_value = self.target_critic_1(next_states,a)
        q2_value = self.target_critic_2(next_states,a)
        # 取出最小的q值  [b, 1]
        # min_qvalue = torch.sum(next_probs * torch.min(q1_value,q2_value), dim=1, keepdims=True)
        min_qvalue = torch.min(q1_value,q2_value)
        # 下个时刻的state_value  [b, 1]
        next_value = min_qvalue + self.log_alpha.exp() * log_pi_a

        # 时序差分，目标网络输出当前时刻的state_value  [b, n_actions]
        td_target = rewards + self.gamma * next_value * (1-dones)
        return td_target
    
    # 软更新，每次训练更新部分参数
    def soft_update(self, net, target_net):
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 预测网络的参数赋给目标网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    # 模型训练
    def update(self, transition_dict):
        mini_batch = random.sample(transition_dict, self.batch_size)
        # 提取数据集
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
  
        # --------------------------------- #
        # 更新2个价值网络
        # --------------------------------- #

        # 目标网络的state_value [b, 1]
        td_target = self.calc_target(rewards, next_states, dones)
        # 价值网络1--预测，当前状态下的动作价值  [b, 1]
        critic_1_qvalues = self.critic_1(states,actions)#.gather(1, actions)
        # 均方差损失 预测-目标
        critic_1_loss = torch.mean(F.mse_loss(critic_1_qvalues, td_target.detach()))
        # 价值网络2--预测
        critic_2_qvalues = self.critic_2(states,actions)#.gather(1, actions)
        # 均方差损失
        critic_2_loss = torch.mean(F.mse_loss(critic_2_qvalues, td_target.detach()))
        
        # 梯度清0
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        # 梯度反传
        critic_1_loss.backward()
        critic_2_loss.backward()
        # 梯度更新
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # --------------------------------- #
        # 更新策略网络
        # --------------------------------- #

        mu, log_std = self.actor(states, deterministic=False, with_logprob=True)  # 预测当前时刻的state_value  [b,n_actions]
        # log_probs = torch.log(probs + 1e-8)  # [b,n_actions]
        # 计算策略网络的熵  [b,1]
        # entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        # 价值网络预测当前时刻的state_value  
        q1_value = self.critic_1(states,mu)  # [b,n_actions]
        q2_value = self.critic_2(states,mu)
        # 取出价值网络输出的最小的state_value  [b,1]
        # min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        min_qvalue = torch.min(q1_value, q2_value)

        # 策略网络的损失
        actor_loss = torch.mean(-self.log_alpha.exp() * log_std - min_qvalue)
        # 梯度更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------------- #
        # 更新可训练遍历alpha
        # --------------------------------- #
        alpha_loss = -(self.log_alpha * (log_std + self.target_entropy).detach()).mean()
			
        # alpha_loss = torch.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        # 梯度更新
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新目标价值网络
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
