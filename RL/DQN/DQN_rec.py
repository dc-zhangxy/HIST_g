# import sys
# import gym
import pylab
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
import os 
import math

# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        return self.fc(x)
    
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
    def __init__(self, action_size=2, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.1, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model+1, action_size)
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


class GRUModel(nn.Module):
    def __init__(self, action_size, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size+1, action_size)

        self.d_feat = d_feat

    def forward(self, x):
        # feature: [N, F*T]
        feature = x[:,:-1]
        xt = x[:,-1:]
        feature = feature.reshape(len(feature), self.d_feat, -1)  # [N, F, T]
        feature = feature.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(feature)
        
        return self.fc_out(torch.concat([out[:, -1, :], xt],1)) #.squeeze() 会把[1,5]的维度变成[5]

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQNAgent, self).__init__()
        # if you want to see Cartpole learning, then change to True
        # self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.ava_action = [j for j in range(action_size)]

        # These are hyper parameters for the DQN
        self.discount_factor = 0.95 #0.99
        # self.learning_rate = 1e-3
        self.memory_size = int(2e4)
        self.epsilon = 0.99 #1.0
        self.epsilon_min = 0.005
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 1000 #64
        self.train_start = 3000 #1000

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)
        self.device = device
        # create main model and target model
        self.model = Transformer(action_size = action_size) # GRUModel(action_size) #  DQN(state_size, action_size) #
        self.target_model = Transformer(action_size = action_size)  # GRUModel(action_size) # DQN(state_size, action_size) #
        self.model.apply(self.weights_init)
        self.model.to(self.device)
        self.target_model.to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                             lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/cartpole_dqn')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #  get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # ava_action = [0,1,2] # torch.nonzero(mask[0]==0).squeeze(1)
            return random.choice(self.ava_action) #self.action_size-1 #  
        else:
            q_value = self.model(state.to(self.device))  #- 1e5*mask
            # print(state.shape, q_value.shape)
            action = torch.max(q_value, 1)[1]  # 在avaliable action 里选择 # 多个标的时再换成 (q_value, 1)[1]
            # print(action)
        return action.cpu()

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        # 将批次数据逐个拆开并存入经验回放池
        # for i in range(state.shape[0]): # batch size
        #     self.memory.append((state[i], action[i], reward[i], next_state[i], done[i]))
        self.memory.append((state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self, optimizer):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)
        # mini_batch = np.array(mini_batch).transpose()
        # 分别解压 mini_batch
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
        
        pred = self.model(states)
        # print(pred.shape, actions.shape)
        pred = pred.gather(1, actions)  # 筛选action位置上的预测值

        # Q function of next state
        next_pred = self.target_model(next_states) # .data  ##?

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0].unsqueeze(1)  # .max(1)[0]
        # target = Variable(target)

        optimizer.zero_grad()

        # MSE Loss function
        # loss = F.mse_loss(pred, target)
        # Compute Huber loss
        loss = F.smooth_l1_loss(pred, target)

        loss.backward()

        # and train
        optimizer.step()
        
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.)

        return loss 