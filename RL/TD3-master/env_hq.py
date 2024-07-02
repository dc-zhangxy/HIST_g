import numpy as np
# import math
import pandas as pd
import torch
import random
# import scipy
from scipy.integrate import quad
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D
import sys
import itertools as it

class HQindexEnv:
    def __init__(self, state_size, action_size, device):
        # self.horizon = horizon
        # self.fare_vec = fare_vec
        # self.n_product = len(fare_vec)
        # self.discount_num = len(discount_list)
        self.device = device 
        # self.inventory_ori = inventory_ori
        # self.inventory = self.inventory_ori.clone()
        
        # self.consumption_mat = torch.ones(n_product)
        
        # self.action_list = action_list
        # self.character_size = character_dim
        # self.n_p = int(self.horizon + self.inventory_ori.sum() + self.character_size )+1
        
        # self.observation_space = spaces.Box(self.n_p, 2*np.ones(self.n_p))
        # self.action_space = spaces.Discrete(len(S_set)*self.discount_num)
        self.state_size = state_size #self.n_p  #self.observation_space.shape[0]
        self.action_size = action_size # len(action_set)*self.discount_num + 1 # self.action_space.n
        # self.shuffle_k = torch.tensor(range(self.horizon))
#         self.character_v = torch.Tensor(character_vector)
        self.tc = 5e-4
        self.xt = 0
        # self.utility_n = np.zeros(self.n_product)
        # self.theta_vec = theta_vec
        # self.theta_x = np.zeros([self.n_product, self.n_product])

        # self.S_phi = dict() #S_phi.copy()
        self.done = False

    def step(self, input_action, label):
        # action \in [-1,1]
        if  self.xt == 0 and input_action > 0:
            # buy
            reward = input_action*((1-self.tc)*label - self.tc)
            self.xt = 1
            
        elif self.xt == 0 and input_action < 0:
            # short , sell or no act
            reward = input_action*label
            
        elif self.xt == 1 and input_action < 0:
            # sell 
            reward = -input_action*(-(1-self.tc)*label - self.tc)
            self.xt = 0
            
        elif self.xt == 1 and input_action > 0:
            # long hold , buy or no act
            reward = input_action*label
            
        # 状态为 过去n日量价 和 当前持仓
        
        return torch.tensor([[self.xt]]), reward #, self.done, .to(self.device)
            
    def _step(self, input_action, label):
        # 0,1  分别代表 short, long 即 -1,1
        if  self.xt == 0 and input_action == 1:
            # buy
            reward = (1-self.tc)*label - self.tc
            self.xt = 1
            
        elif self.xt == 0 and input_action == 0:
            # short , sell or no act
            reward = -label
            
        elif self.xt == 1 and input_action == 0:
            # sell 
            reward = -(1-self.tc)*label - self.tc
            self.xt = 0
            
        elif self.xt == 1 and input_action == 1:
            # long hold , buy or no act
            reward = label
            
        # 状态为 过去n日量价 和 当前持仓
        
        return torch.tensor([[self.xt]]), reward #, self.done, .to(self.device)
    
    def __step(self, input_action, label):
        # 0,1,2 分别代表 buy,noact,sell,即 1,0,-1
        if  self.xt==0 and input_action == 0:
            # buy
            reward = (1-self.tc)*label - self.tc
            self.xt = 1
            
        elif self.xt ==0 and input_action in [2,1]:
            # short , sell or no act
            reward = -label
            
        elif self.xt==1 and input_action == 2:
            # sell # reward = -(1-self.tc)*label ??
            # reward = (1-self.tc)*(1-label) -1
            # reward = (1-self.tc)*label - 2* self.tc 
            reward = -(1-self.tc)*label - self.tc
            self.xt = 0
            
        elif self.xt==1 and input_action in [0,1]:
            # long hold , buy or no act
            reward = label
            
        # 状态为 过去n日量价 和 当前持仓
        
        return torch.tensor([[self.xt]]), reward #, self.done, .to(self.device)
            
    def reset(self, e):
        # np.random.seed(e)
        self.done = False
        # self.steps = 0
        # self.inventory = self.inventory_ori.clone()

        self.xt = 0

        return torch.tensor([[self.xt]]) #.to(self.device)
    # def fun_demand(self, bundle, discount, e):
        
    #     return purchase_bundle, torch.tensor(purchase_list)
            