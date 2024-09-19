import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
# import qlib
from torch.utils.tensorboard import SummaryWriter
# from qlib.contrib.model.pytorch_gru import GRUModel
# from qlib.contrib.model.pytorch_transformer import Transformer
# from pytorch_transformer import Transformer
from utils import metric_fn, mse
from dataloader import DataLoader
from env_hq import HQindexEnv
# from DQN_rec import DQNAgent 
# import TD3s 
import PPO as PPO 
# import OurDDPG
# import DDPG

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12

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

global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)
'''
# global_step = -1
def train_epoch(epoch, optimizer, agent, env, train_loader, train_data_dt_st, writer, args):
    # print(epoch)
    done = False
    revenue_sum = 0
    xt = env.reset(epoch)
    # agent.train()
    for st in train_data_dt_st['instrument'].drop_duplicates():
        revenue_list = []
        revenue = 0
        train_data_st = train_data_dt_st[train_data_dt_st['instrument']==st].reset_index(drop=True)
        train_data_st_y = train_data_st.copy() #train_data_st[train_data_st['datetime'].isin(yeardatelist)].reset_index(drop=True)
        train_end_index = len(train_data_st_y)
        for dt_ind in range(train_end_index):
            st_date = train_data_st_y.loc[dt_ind,'datetime']
            feature, label, = train_loader.get(st_date, st)
            state = torch.concat([feature, xt],-1) 
            # Select action randomly or according to policy
            if epoch < args.start_timesteps:
                # print('random action')
                action = (torch.rand(1,1)*2*args.max_action - args.max_action).data.numpy()  #torch.tensor([random.uniform(-args.max_action, args.max_action)])
            else:
                action = (
                    agent.get_action(state)  #(np.array(state))
                    + np.random.normal(0, args.max_action * args.expl_noise, size=args.action_size)
                ).clip(-args.max_action, args.max_action)
                
            xt, reward = env.step(action, label)
            revenue_list.append(reward[0][0])
            # if len(revenue_list)>1:
            #     bdl = pd.DataFrame(revenue_list).rolling(180,min_periods=1).std().iloc[-1,0]
            #     # print(reward, bdl)
            #     if bdl !=0:
            #         reward = reward/bdl
            revenue += reward
            
            if dt_ind < train_end_index - 1:
                st_date_1 = train_data_st_y.loc[dt_ind+1,'datetime']
                feature, label, = train_loader.get(st_date_1, st)
            else :
                done = True 
                st_date_0 = train_data_st_y.loc[0,'datetime']
                feature, label,  = train_loader.get(st_date_0, st)
                
            next_state = torch.concat([feature, xt],-1) 
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            
            if epoch >= args.start_timesteps:  # len(agent.memory) >= agent.train_start:
                agent.train_model()
        revenue_sum += revenue
        return revenue_sum
'''

def test_epoch(epoch, agent, env, test_loader, train_data_dt_st, writer, args, prefix='Test'):
    # agent.eval()
    losses = [0]
    preds = []
    revenue_sum = 0
    xt = env.reset(epoch)
    # for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
    for st in train_data_dt_st['instrument'].drop_duplicates():
        revenue = 0
        revenue_list = []
        train_data_st = train_data_dt_st[train_data_dt_st['instrument']==st].reset_index(drop=True)
        # train_data_st['year'] = train_data_st['datetime'].dt.year
        yearly_groups = [1]  # train_data_st.groupby('year')['datetime'].apply(list)
        # print(yearly_groups)
        for yeardatelist in yearly_groups:
            train_data_st_y = train_data_st.copy() #train_data_st[train_data_st['datetime'].isin(yeardatelist)].reset_index(drop=True)
            # print(train_data_st_y)
            train_end_index = len(train_data_st_y)
            for dt_ind in range(train_end_index):
                st_date = train_data_st_y.loc[dt_ind,'datetime']
                feature, label, = test_loader.get(st_date, st)
                with torch.no_grad():
                    # state = [feature, xt]
                    state = torch.concat([feature, xt],-1) 
                    # if (state.isnan().any()):
                    #     print('T')
                    action, probs = agent.take_action(state)  #(np.array(state))
                    xt, reward = env.step(action, label)
                    revenue_list.append(reward)
                    revenue += reward
                if prefix=='Test':
                    # preds_df = pd.DataFrame({ 'action': action,'probs0':probs[:,0].item(),'probs1':probs[:,1].item(), 'label': label.cpu().detach().numpy(), 'datetime': st_date ,'instrument':st})
                    preds_df = pd.DataFrame({ 'action': action,'probs1':probs, 'label': label.cpu().detach().numpy(), 'datetime': st_date ,'instrument':st})
                    preds.append(preds_df.set_index(['datetime','instrument'],drop=True))
        revenue_sum += revenue
    if prefix=='Test':
        preds = pd.concat(preds, axis=0)
    return np.mean(losses), revenue_sum, preds #, precision, recall, ic, rank_ic

def inference(agent, env, data_loader, train_data_dt_st):
    # agent.eval()
    revenue_sum = 0
    xt = env.reset(0)
    preds = []
    # for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
    for st in train_data_dt_st['instrument'].drop_duplicates():
        revenue = 0
        revenue_list = []
        train_data_st = train_data_dt_st[train_data_dt_st['instrument']==st].reset_index(drop=True)
        # train_data_st['year'] = train_data_st['datetime'].dt.year
        yearly_groups = [1]  # train_data_st.groupby('year')['datetime'].apply(list)
        # print(yearly_groups)
        for yeardatelist in yearly_groups:
            train_data_st_y = train_data_st.copy() #train_data_st[train_data_st['datetime'].isin(yeardatelist)].reset_index(drop=True)
            # print(train_data_st_y)
            train_end_index = len(train_data_st_y)
            for dt_ind in range(train_end_index):
                st_date = train_data_st_y.loc[dt_ind,'datetime']
                feature, label, = data_loader.get(st_date, st)
                with torch.no_grad():
                    state = torch.concat([feature, xt],-1) 
                    action,probs = agent.take_action(state)  #(np.array(state))
                    xt, reward = env.step(action, label)
                    revenue_list.append(reward)
                    revenue += reward
                # preds_df = pd.DataFrame({ 'action': action, 'probs0':probs[:,0].item(),'probs1':probs[:,1].item(),'label': label.cpu().detach().numpy(), 'datetime': st_date ,'instrument':st})
                preds_df = pd.DataFrame({ 'action': action,'probs1':probs, 'label': label.cpu().detach().numpy(), 'datetime': st_date ,'instrument':st})
                preds.append(preds_df.set_index(['datetime','instrument'],drop=True))
        revenue_sum += revenue
    preds = pd.concat(preds, axis=0)
    return preds, revenue_sum

def create_loaders(args, train_data, val_data, test_data ):

    Feature = args.features # ['s_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 's_dq_adjclose', 's_dq_volume',] # 's_dq_amount']
    Columns = []
    # 原始方法：除以当天close
    for f in Feature:
        for i in range(args.days-1,0,-1):
            Columns.append(f+'_lag_'+str(i))
        Columns.append(f)

    dev_train = train_data.set_index(['datetime', 'instrument'])
    dev_valid = val_data.set_index(['datetime', 'instrument'])
    dev_test_mul = test_data.set_index(['datetime', 'instrument'])
    
    start_index = 0

    train_loader = DataLoader(dev_train[Columns], dev_train['label'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    start_index += len(dev_train.groupby(level=0,group_keys=False).size())

    valid_loader = DataLoader(dev_valid[Columns], dev_valid['label'], pin_memory=False, start_index=start_index, device = device)
    
    start_index += len(dev_valid.groupby(level=0,group_keys=False).size())

    test_loader = DataLoader(dev_test_mul[Columns], dev_test_mul['label'], pin_memory=False, start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader, train_data[['datetime', 'instrument']], val_data[['datetime', 'instrument']],test_data[['datetime', 'instrument']]

def main(args, train_data, val_data, test_data ):
    seed = args.seed  # np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')

    start_dates=[]
    start_dates.append(args.train_start_date)
    train_loaders = []
    valid_loaders = []
    test_loaders = []
    train_lens = []
    valid_lens = []
    test_lens = []
    for start_date in start_dates:
        pprint('start loading', start_date)
        train, valid, test, train_len, valid_len, test_len = create_loaders(args, train_data, val_data, test_data )
        pprint(start_date, ' done...in progress...')
        train_loaders.append(train)
        train_lens.append(train_len)
        valid_loaders.append(valid)
        valid_lens.append(valid_len)
        test_loaders.append(test)
        test_lens.append(test_len)
    # train_loader, valid_loader, test_loader = create_loaders(args)
    pprint('loaders done')
    out_put_path_base = output_path

    for i in range(0,len(train_loaders)):
        train_loader = train_loaders[i]
        train_ilens = train_lens[i]
        test_loader = test_loaders[i]
        test_ilens = test_lens[i]
        valid_loader = valid_loaders[i]
        valid_ilens = valid_lens[i]
        output_path = out_put_path_base +'/rolling' + start_dates[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # all_precision = []
        # all_recall = []
        # all_ic = []
        # all_rank_ic = []
        steps = 0
        for times in range(args.repeat):
            pprint('create model...')
            # if args.model_name == 'TRANSFORMER':
            #     model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=args.dropout)
            # else:
            #     model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)
            state_size = args.days * args.d_feat +1
            action_size = args.action_size
            env = HQindexEnv(state_size, action_size, device)
    
            # state_size = env.state_size  # env.observation_space.shape[0]
            # action_size = env.action_size  # env.action_space.n
            # model = DQNAgent(state_size, action_size, device)
            # model.to(device)
            kwargs = {
            "state_dim": state_size,
            "action_dim": action_size,
            "max_action": 1,
            # "discount": args.discount,
            # "tau": args.tau,
            "device": device
            }
            # Initialize policy
            if args.policy == "PPO":
                # Target policy smoothing is scaled wrt the action scale
                # kwargs["policy_noise"] = args.policy_noise * args.max_action
                # kwargs["noise_clip"] = args.noise_clip * args.max_action
                # kwargs["policy_freq"] = args.policy_freq
                model = PPO.PPO(**kwargs)
            # elif args.policy == "OurDDPG":
            #     model = OurDDPG.DDPG(**kwargs)
            # elif args.policy == "DDPG":
            #     model = DDPG.DDPG(**kwargs)

            optimizer = None #optim.Adam(model.parameters(), lr=args.lr)
            
            # 定义学习率调度器
            # Re_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=25, verbose=True)
            best_score = -np.inf
            best_epoch = 0
            # epochs_no_improve = 0
            stop_round = 0
            best_param = copy.deepcopy(model.actor.state_dict())
            params_list = collections.deque(maxlen=args.smooth_steps)

            train_revenue_list = []
            val_score_list = []
            test_score_list = []
            # create replay memory using deque
            # memory_size = int(1e6)
            # memory = [] # collections.deque()
            for epoch in range(args.n_EPISODES):
                pprint('Running', times,'Epoch:', epoch)
                pprint('training...')
                # train_revenue = train_epoch(epoch, optimizer, model, env, train_loader, train_ilens, writer, args)
            # global_step = -1 
            # def train_epoch(epoch, optimizer, model, env, train_loader, train_ilens, writer, args):
                # print(epoch)
                done = False
                revenue_sum = 0
                xt = env.reset(epoch)
                # agent.train()
                for st in train_ilens['instrument'].drop_duplicates():
                    memory = []
                    revenue = 0
                    revenue_list = []
                    train_data_st = train_ilens[train_ilens['instrument']==st].reset_index(drop=True)
                    train_data_st_y = train_data_st.copy() #train_data_st[train_data_st['datetime'].isin(yeardatelist)].reset_index(drop=True)
                    train_end_index = len(train_data_st_y)
                    for dt_ind in range(train_end_index):
                        steps += 1
                        st_date = train_data_st_y.loc[dt_ind,'datetime']
                        feature, label, = train_loader.get(st_date, st)
                        state = torch.concat([feature, xt],-1) 
                        # Select action
                        action, probs = model.take_action(state)  #(np.array(state))
                        xt, reward = env.step(action, label)
                        revenue_list.append(reward)
                        revenue += reward
                        
                        if dt_ind < train_end_index - 1:
                            st_date_1 = train_data_st_y.loc[dt_ind+1,'datetime']
                            feature, label, = train_loader.get(st_date_1, st)
                        else :
                            done = True 
                            st_date_0 = train_data_st_y.loc[0,'datetime']
                            feature, label,  = train_loader.get(st_date_0, st)
                            
                        next_state = torch.concat([feature, xt],-1) 
                        # save the sample <s, a, r, s'> to the replay memory
                        # model.append_sample(state, action, reward, next_state, done)
                        memory.append((state, action, reward, next_state, done))
                    revenue_sum += revenue
                    train_revenue_list.append([epoch,steps,revenue_sum])
                    pd.DataFrame(train_revenue_list,columns=['epoch','steps','score']).to_pickle(output_path+'/train_revenue_list.pkl')
                        # if epoch >= args.start_timesteps:  # len(agent.memory) >= agent.train_start:
                    model.learn(memory)
                    # if steps%10 ==0:
                pprint('evaluating...')
                pprint('epoch:', epoch, 'steps:', steps)
                val_loss, val_score, val_preds = test_epoch(epoch, model, env, valid_loader, valid_ilens, writer, args, prefix='Valid')
                test_loss, test_score, test_preds = test_epoch(epoch, model, env, test_loader, test_ilens, writer, args, prefix='Test')
                val_score_list.append([epoch,steps, val_score])
                test_score_list.append([epoch,steps,test_score])
                test_preds.to_pickle(output_path+'/pred.pkl.'+'test'+str(times)+str(epoch)+str(steps))
                pd.DataFrame(val_score_list,columns=['epoch','steps','score']).to_pickle(output_path+'/val_score_list.pkl')
                pd.DataFrame(test_score_list,columns=['epoch','steps','score']).to_pickle(output_path+'/test_score_list.pkl')
                
                # pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
                # pprint(' valid_score %.6f, test_score %.6f'%( val_score, test_score))
                            
            # if epoch >= args.start_timesteps and steps%10 ==0:
                pprint('evaluating...')
                pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(revenue_sum, val_score, test_score))
                # model.actor.load_state_dict(params_ckpt)
                # 使用调度器调整学习率
                # Re_scheduler.step(-val_score)
                # 至少训练10轮
                if epoch >= 5:
                    if val_score > best_score:
                        best_score = val_score
                        stop_round = 0
                        # epochs_no_improve = 0
                        best_epoch = epoch
                        best_param = copy.deepcopy(model.actor.state_dict())
                        # best_param = copy.deepcopy(avg_params)
                    else:
                        stop_round += 1
                        if stop_round >= args.early_stop:
                            pprint('early stop')
                            break
                    pprint('Best epoch: ', best_epoch)
                
            pprint('best score:', best_score, '@', best_epoch)
            
            # model.actor.load_state_dict(best_param)
            # torch.save(best_param, output_path+'/model.bin')

            # pprint('inference...')
            res = dict()
            # for name in ['test']:
            #     pred, revenue= inference(model, env, eval(name+'_loader'), eval(name+'_ilens'))
            #     pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            pprint('save info...')

            info = dict(
                config=vars(args),
                best_epoch=best_epoch,
                best_score=res,
            )
            default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
            with open(output_path+'/info.json', 'w') as f:
                json.dump(info, f, default=default, indent=4)

    pprint('finished.')


class ParseConfigFile(argparse.Action):
    def __call__(self, parser, namespace, filename, option_string=None):
        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)
        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def data_prepare(mode, args):
    # import pandas as pd
    test = pd.read_csv(args.filepath, index_col=0)
    test['END_DATE'] = pd.to_datetime(test['END_DATE'])
    test = test[test['STOCK_CODE'] =='000300'].reset_index(drop=True )
    # test = test[test['STOCK_CODE'].isin(['000852','000905','000300','000001'])].reset_index(drop=True )
    # 如果要跑全市场的，
    test = test.sort_values(by=['END_DATE', 'STOCK_CODE'])
        
    test['PRE_CLOSE'] = test.groupby('STOCK_CODE',group_keys=False)['CLOSE'].shift(-args.labels)
    test['NEXT_CLOSE'] = test.groupby('STOCK_CODE',group_keys=False)['CLOSE'].shift(-args.NEXT_CLOSE)
    # 创建1日标签
    test['label'] = test['PRE_CLOSE'].div( test['NEXT_CLOSE'], axis=0)-1
    # test['label_std'] = test.groupby('STOCK_CODE',group_keys=False)['label'].rolling(180).std().values
    # test['sharp_label'] = test['label'].div(test['label_std'] ,axis=0)
    
    test = test.drop(['PRE_CLOSE', 'NEXT_CLOSE'],axis=1)
    test1 = test.dropna().reset_index(drop=True) #.drop('Unnamed: 0',axis=1)
    # .to_csv('test1.csv',index=False)

    # 所有特征列，除了'STOCK_CODE'和'END_DATE'
    feature_columns = args.features # ['s_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 's_dq_adjclose', 's_dq_volume', ] #'s_dq_amount']

    # 定义一个函数，生成过去60天的特征数据
    # 我这里是直接shift的，所以对于前800数量有变化的股票就会有时间断层，需要保留当前时间点前60天的数据才行
    def generate_past_60_days_features(group, features, days=args.days):
        new_columns = []
        for feature in features:
            for i in range(1, days + 1):
                new_columns.append(group[feature].shift(i).rename(f'{feature}_lag_{i}'))
        # 使用pd.concat(axis=1)一次性连接所有列
        new_data = pd.concat(new_columns, axis=1)
        # 将新生成的列添加到原DataFrame中
        group = pd.concat([group, new_data], axis=1)
        return group

    # 使用group.copy()创建副本，并应用函数
    df_cleaned = test1.groupby('STOCK_CODE',group_keys=False).apply(lambda x: generate_past_60_days_features(x.copy(), features = feature_columns))

    # 去除前60天的NaN数据
    expanded_df = df_cleaned.dropna().reset_index(drop=True)

    # 标准化特征
    Feature = ['OPEN', 'HIGH', 'LOW', 'CLOSE',] # , ['VOL', 'VALUE']
    Columns_c = ['OPEN', 'HIGH', 'LOW', 'CLOSE',]  # 需要除以close的特征
    
    for i in range(1,args.days+1):
        for f in Feature:
            Columns_c.append(f+'_lag_'+str(i))
    expanded_df[Columns_c] = expanded_df[Columns_c].div( expanded_df['CLOSE'], axis=0)-1

    # Feature = ['VOL', 'VALUE'] # 需要除以vol和value以归一化的特征
    Columns_vol = ['VOL' ]
    Columns_value = ['VALUE']
    for i in range(1,args.days+1):
        Columns_vol.append('VOL'+'_lag_'+str(i))
        Columns_value.append('VALUE'+'_lag_'+str(i))
            
    expanded_df[Columns_vol] = expanded_df[Columns_vol].div( expanded_df['VOL'], axis=0) -1
    expanded_df[Columns_value] = expanded_df[Columns_value].div( expanded_df['VALUE'], axis=0) -1
    
    dev_test = expanded_df.dropna().reset_index(drop=True)
    
    #.to_csv('dev_test.csv', index=False)
    dev_test  = dev_test.rename(columns={"STOCK_CODE": "instrument", "END_DATE": "datetime"})
    
    train = dev_test[(dev_test['datetime'] >= args.train_start_date) & (dev_test['datetime'] <= args.train_end_date)]
    val = dev_test[(dev_test['datetime'] >= args.valid_start_date) & (dev_test['datetime'] <= args.valid_end_date)]
    test = dev_test[(dev_test['datetime'] >= args.test_start_date) & (dev_test['datetime'] <= args.test_end_date)]

    # dev_test = dev_test.set_index(['datetime', 'instrument']).sort_index(level=['datetime', 'instrument'])
    # dev_test
    # dev_test.to_csv('dev_'+mode+'_mul'+str(args.labels)+'_'+str(args.days)+'.csv')
    return train, val, test


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='GRU') # GRU TRANSFORMER
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--action_size', type=int, default=1)  # 2
    parser.add_argument('--d_feat', type=int, default=6)  # 6
    parser.add_argument('--features', default= 
        ['OPEN', 'HIGH', 'LOW', 'CLOSE','VOL', 'VALUE'])  # 6
    
    parser.add_argument("--policy", default="PPO")                  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="hq")          # OpenAI gym environment name # HalfCheetah-v2
    parser.add_argument("--seed", default=100, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1, type=int)# Time steps initial random policy is used # 25e3
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--max_action", default=1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--discount", default=0.95, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    # training
    parser.add_argument('--n_EPISODES', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=250)
    parser.add_argument('--smooth_steps', type=int, default=5)
    # parser.add_argument('--actor_lr_decay_step', type=int, default=50)
    
    # parser.add_argument('--metric', default='loss')
    parser.add_argument('--loss', default='mse') # mse CE
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--filepath', type=str, default='../data/hq_index.csv') #'../data/hq_index.csv'
    # /home/aiuser/work/HIST_all/RL/data/hq_index.csv
    # /home/aiuser/work/HIST_all/RL/

    # data
    parser.add_argument('--data_set', type=str, default='all')
    parser.add_argument('--pin_memory', action='store_false', default=False)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    
    parser.add_argument('--train_start_date', default='2009-01-01') #1990-12-19
    parser.add_argument('--train_end_date', default='2016-12-31') # 2017-12-31
    parser.add_argument('--valid_start_date', default='2017-01-01') # 2018-01-01
    parser.add_argument('--valid_end_date', default='2018-12-31') # 2019-12-31
    parser.add_argument('--test_start_date', default='2019-01-01') # 2020-01-01
    parser.add_argument('--test_end_date', default='2024-06-18') # 2024-06-18
    parser.add_argument('--labels', type=int, default=2)
    parser.add_argument('--NEXT_CLOSE', type=int, default=1)
    parser.add_argument('--days', type=int, default=30) # 30 60 

    # other
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='all_transf') # 07_20

    parser.add_argument('--outdir', default='./output/hq300_PPO_transf1_label2to1_1prob')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    pprint('begin...')
    # 数据预处理
    train, val, test = data_prepare('train',args)
    # train.to_csv('data/train.csv', index=False)
    # val.to_csv('val.csv', index=False)
    # data_prepare('../test.txt','test',args)
    pprint('data prepare done...')

    if args.mode == 'train':
        main(args, train, val ,test)
    elif args.mode == 'test':
        pprint('loadandinference!')
        # loadandinference(args)
    else:
        pprint('mode name error!')
