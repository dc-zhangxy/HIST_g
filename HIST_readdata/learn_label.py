import torch
# import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
# import qlib
# regiodatetimeG_CN, REG_US]
# from dateutil.relativedelta import relativedelta
# from qlib.config import REG_US, REG_CN
# from dateutil.relativedelta import relativedelta
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# provider_uri = "../data/cn_data_updated"  # target_dir
# qlib.init(provider_uri=provider_uri, region=REG_CN)
# from qlib.data.dataset import DatasetH
# from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
# from qlib.contrib.model.pytorch_lstm import LSTMModel
# from qlib.contrib.model.pytorch_gats import GATModel
# from qlib.contrib.model.pytorch_sfm import SFM_Model
# from qlib.contrib.model.pytorch_alstm import ALSTMModel
# from qlib.contrib.model.pytorch_transformer import Transformer
from model import MLP, HIST
from utils import metric_fn, mse
from dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP
    
    if model_name.upper() == 'GRU':
        return GRUModel

    if model_name.upper() == 'HIST':
        return HIST

    raise ValueError('unknown model name `%s`'%model_name)


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



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])


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


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concepts = None):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value , stock_index, index = train_loader.get(slc)
        # print(feature.shape, label.shape, market_value.shape )
        # print(feature)
        # print(label)
        # print(market_value)
        # print(stock_index)
        # print(index)
        # 考虑公告期
        if index[0][0] <= datetime.datetime(2009,4,30): # 第一个数据是20081231因此只能在20090430之后使用
            stock2concept_matrix = torch.zeros([6000,5]).to(device)
        else:
            if index[0][0] <= datetime.datetime(2010,8,30): # 2010半年报之前都只有年报
                # 这里有点问题因为还需要考虑430之后才能使用年报的问题，应修改
                if index[0][0].month > 4: # 430之后才能使用去年的年报
                    selection = str(index[0][0].year-1) +'1231'
                else:                     # 430之前都只能使用前年的年报
                                        # 可是这样的话就只能使用20090430之后的数据训练
                    selection = str(index[0][0].year-2) +'1231'
                    
            elif index[0][0].month <= 4:  # 430年报披露，在此之前使用上一年的半年报
                selection = str(index[0][0].year-1) +'0630'
            elif index[0][0].month <= 8: #830半年报披露，在此之前使用上一年的年报
                selection = str(index[0][0].year-1) +'1231'
            else:                       # 830之后使用当年披露的半年报
                selection = str(index[0][0].year) +'0630'

            stock2concept_matrix = torch.Tensor(stock2concepts[int(selection)]).to(device)
            
        if args.model_name == 'HIST':
            # print(stock_index)
            # print(type(stock_index))
            # print(stock2concept_matrix[stock_index])
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
        else:
            pred = model(feature)
            
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concepts=None, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)
        # 考虑公告期
        if index[0][0] <= datetime.datetime(2009,4,30): # 第一个数据是20081231因此只能在20090430之后使用
            stock2concept_matrix = torch.zeros([6000,5]).to(device)
        else:
            if index[0][0] <= datetime.datetime(2010,8,30): # 2010半年报之前都只有年报
                # 这里有点问题因为还需要考虑430之后才能使用年报的问题，应修改
                if index[0][0].month > 4: # 430之后才能使用去年的年报
                    selection = str(index[0][0].year-1) +'1231'
                else:                     # 430之前都只能使用前年的年报
                                        # 可是这样的话就只能使用20090430之后的数据训练
                    selection = str(index[0][0].year-2) +'1231'
                    
            elif index[0][0].month <= 4:  # 430年报披露，在此之前使用上一年的半年报
                selection = str(index[0][0].year-1) +'0630'
            elif index[0][0].month <= 8: #830半年报披露，在此之前使用上一年的年报
                selection = str(index[0][0].year-1) +'1231'
            else:                       # 830之后使用当年披露的半年报
                selection = str(index[0][0].year) +'0630'

            stock2concept_matrix = torch.Tensor(stock2concepts[int(selection)]).to(device)
            
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            else:
                pred = model(feature)
                
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
            # print(preds)
            # print(index)
            
        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(model, data_loader, stock2concepts=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        # 考虑公告期
        if index[0][0] <= datetime.datetime(2009,4,30): # 第一个数据是20081231因此只能在20090430之后使用
            stock2concept_matrix = torch.zeros([6000,5]).to(device)
        else:
            if index[0][0] <= datetime.datetime(2010,8,30): # 2010半年报之前都只有年报
                # 这里有点问题因为还需要考虑430之后才能使用年报的问题，应修改
                if index[0][0].month > 4: # 430之后才能使用去年的年报
                    selection = str(index[0][0].year-1) +'1231'
                else:                     # 430之前都只能使用前年的年报
                                        # 可是这样的话就只能使用20090430之后的数据训练
                    selection = str(index[0][0].year-2) +'1231'
                    
            elif index[0][0].month <= 4:  # 430年报披露，在此之前使用上一年的半年报
                selection = str(index[0][0].year-1) +'0630'
            elif index[0][0].month <= 8: #830半年报披露，在此之前使用上一年的年报
                selection = str(index[0][0].year-1) +'1231'
            else:                       # 830之后使用当年披露的半年报
                selection = str(index[0][0].year) +'0630'

            stock2concept_matrix = torch.Tensor(stock2concepts[int(selection)]).to(device)
            
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            else:
                pred = model(feature)
                
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds

# 得到两个日期之间的年月序列，方便读取数据
def get_year_month(start_date,end_date):
    start_year, start_month = start_date.year, start_date.month
    end_year, end_month = end_date.year, end_date.month
    interval=(end_year - start_year)*12 + (end_month - start_month)
    cur_ym = []
    for i in range(interval):
        now_month = start_month + i
        year_tmp, month_tmp = start_year + now_month // 12, now_month % 12
        if month_tmp == 0:
            month_tmp = 12
            year_tmp -= 1
        cur_date = datetime.datetime.strptime(f'{year_tmp}-{month_tmp}', '%Y-%m')
        interval_date = cur_date.strftime('%Y-%m')
        cur_ym.append(interval_date)
    return cur_ym

def create_loaders(args,train_start_date):
    train_start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    valid_start_time = datetime.datetime.strptime(args.valid_start_date, '%Y-%m-%d')
    valid_end_time = datetime.datetime.strptime(args.valid_end_date, '%Y-%m-%d')
    test_start_time = datetime.datetime.strptime(args.test_start_date, '%Y-%m-%d')
    test_end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    
    train_time_list = get_year_month(train_start_time,train_end_time)
    valid_time_list = get_year_month(valid_start_time,valid_end_time)
    test_time_list = get_year_month(test_start_time,test_end_time)
    # hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
    #             'label': [f'Ref($close,{-args.labels}) / Ref($close,-1) - 1']}}
    # segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    # dataset = DatasetH(hanlder,segments)
    # print(f'train: ({args.train_start_date}, {args.train_end_date}), valid: ({args.valid_start_date}, {args.valid_end_date}), test: ({args.test_start_date}, {args.test_end_date})')
    feature = args.feature_list #['close','high','low','open','volume','vwap']
    
    df_train = pd.DataFrame()
    for y in train_time_list:
        c0 = pd.read_feather(args.feature_path + feature[0] +'_'+str(y)+'.feather')
        # print(y, c0.shape)
        for fe in feature[1:]:
            h0 = pd.read_feather(args.feature_path + fe +'_'+str(y)+'.feather')
            c0 = pd.merge(c0,h0, on = ['end_date','stock_code'])
            # print(fe, c0.shape)
        df_train = pd.concat([df_train, c0])
        # print(df_train.shape)
    
        
    df_valid = pd.DataFrame()
    for y in valid_time_list:
        c0 = pd.read_feather(args.feature_path + feature[0] +'_'+str(y)+'.feather')
        for fe in feature[1:]:
            h0 = pd.read_feather(args.feature_path + fe +'_'+str(y)+'.feather')
            c0 = pd.merge(c0,h0, on = ['end_date','stock_code'])
        df_valid = pd.concat([df_valid, c0])
    # print(df_valid)
        
    df_test = pd.DataFrame()
    for y in test_time_list:
        c0 = pd.read_feather(args.feature_path + feature[0] +'_'+str(y)+'.feather')
        for fe in feature[1:]:
            h0 = pd.read_feather(args.feature_path + fe +'_'+str(y)+'.feather')
            c0 = pd.merge(c0,h0, on = ['end_date','stock_code'])
        df_test = pd.concat([df_test, c0])
    # print(df_test)
            
    # df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    
    # with open('data2/df_train09_16.pkl', 'wb') as f:
    #     pickle.dump(df_train, f)
    # with open('data2/df_valid17_18.pkl', 'wb') as f:
    #     pickle.dump(df_valid, f)
    # with open('data2/df_test19_22.pkl', 'wb') as f:
    #     pickle.dump(df_test, f)
    
    # import pickle5 as pickle
    # with open(args.market_value_path, "rb") as fh:
    #     df_market_value = pickle.load(fh)
    df_test['end_date'] = pd.to_datetime(df_test['end_date']) 
    df_valid['end_date'] = pd.to_datetime(df_valid['end_date']) 
    df_train['end_date'] = pd.to_datetime(df_train['end_date']) 
    
    df_close = pd.read_pickle(args.close_path).rename(columns={'trade_dt':'end_date', 's_info_windcode':'stock_code', 's_dq_adjclose': 'Close'})
    df_close['end_date'] = pd.to_datetime(df_close['end_date']) 
    
    def calculate_future_returns(group, label_days=args.labels, begin_days = args.begin_days):
        group['label'] = group['Close'].shift(-label_days).div(group['Close'].shift(-begin_days), axis=0) - 1
        return group
    
    df_close = df_close.sort_values(['end_date','stock_code'])
    label_df = df_close.groupby('stock_code', group_keys=False).apply(calculate_future_returns)
    label_df = label_df.drop(['Close'],axis=1).dropna().reset_index(drop=True)

    # CSRankNorm
    label_df = label_df.set_index(['end_date','stock_code'])
    xx = label_df['label'].groupby('end_date').rank(pct = True)
    xx2 = xx.groupby(level=0, group_keys=False).apply(lambda x:(x-x.mean())/x.std())
    label_df = xx2.reset_index()
    
    df_train = pd.merge(df_train, label_df, on = ['end_date','stock_code']).rename(columns={'end_date':'datetime', 'stock_code':'instrument'}).set_index(['datetime','instrument'])
    df_valid = pd.merge(df_valid, label_df, on = ['end_date','stock_code']).rename(columns={'end_date':'datetime', 'stock_code':'instrument'}).set_index(['datetime','instrument'])
    df_test = pd.merge(df_test, label_df, on = ['end_date','stock_code']).rename(columns={'end_date':'datetime', 'stock_code':'instrument'}).set_index(['datetime','instrument'])
        
    df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    
    stock_index = np.load(args.stock_index_path, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    # slc = slice(args.train_start_date, args.train_end_date)
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train.iloc[:,:-3].astype(np.float32), df_train["label"].astype(np.float32), df_train['market_value'].astype(np.float32), df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    # slc = slice(args.valid_start_date, args.valid_end_date)
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid.iloc[:,:-3].astype(np.float32), df_valid["label"].astype(np.float32), df_valid['market_value'].astype(np.float32), df_valid['stock_index'], pin_memory=False, start_index=start_index, device = device)
    
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    # slc = slice(args.test_start_date, args.test_end_date)
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test.iloc[:,:-3].astype(np.float32), df_test["label"].astype(np.float32), df_test['market_value'].astype(np.float32), df_test['stock_index'], pin_memory=False, start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader


def loadandinference(args):
    pprint('create model...')
    if args.model_name == 'SFM':
        model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
    elif args.model_name == 'ALSTM':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
    elif args.model_name == 'GCN':
        model = get_model(args.model_name)()
    else:
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)

    # model.load_state_dict
    model.to(device)
    output_path = args.outdir
    best_param = output_path +'/rolling' + '2007-01-01' + '/model.bin'
    model.load_state_dict(torch.load(best_param))
    # model.eval()

    pprint('loading data...')
    newdataloader = create_newloader(args)
    name = 'newdata'
    
    with open(args.stock2concept_matrix_path,'rb') as f:
        stock2concepts=pickle.load(f)

    pprint('inference...')

    pred = inference(model, newdataloader, stock2concepts)
    pred.to_pickle(output_path+'/newdata_pred.pkl')

    precision, recall, ic, rank_ic = metric_fn(pred)

    pprint(('%s: IC %.6f Rank IC %.6f')%(
                name, ic.mean(), rank_ic.mean()))
    pprint(name, ': Precision ', precision)
    pprint(name, ': Recall ', recall)


def create_newloader(args):
    train_start_time = datetime.datetime.strptime(args.new_start_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.new_end_date, '%Y-%m-%d')

    train_time_list = get_year_month(train_start_time,train_end_time)

    feature = args.feature_list #['close','high','low','open','volume','vwap']
    
    df_train = pd.DataFrame()
    for y in train_time_list:
        c0 = pd.read_feather(args.feature_path + feature[0] +'_'+str(y)+'.feather')
        # print(y, c0.shape)
        for fe in feature[1:]:
            h0 = pd.read_feather(args.feature_path + fe +'_'+str(y)+'.feather')
            c0 = pd.merge(c0,h0, on = ['end_date','stock_code'])
            # print(fe, c0.shape)
        df_train = pd.concat([df_train, c0])
        # print(df_train.shape)
    df_train['end_date'] = pd.to_datetime(df_train['end_date']) 
    
    df_close = pd.read_pickle(args.close_path).rename(columns={'trade_dt':'end_date', 's_info_windcode':'stock_code', 's_dq_adjclose': 'Close'})
    df_close['end_date'] = pd.to_datetime(df_close['end_date']) 
    
    def calculate_future_returns(group, label_days=args.labels, begin_days = args.begin_days):
        group['label'] = group['Close'].shift(-label_days).div(group['Close'].shift(-begin_days), axis=0) - 1
        return group
    
    df_close = df_close.sort_values(['end_date','stock_code'])
    label_df = df_close.groupby('stock_code', group_keys=False).apply(calculate_future_returns)
    label_df = label_df.drop(['Close'],axis=1).dropna().reset_index(drop=True)

    # CSRankNorm
    label_df = label_df.set_index(['end_date','stock_code'])
    xx = label_df['label'].groupby('end_date').rank(pct = True)
    xx2 = xx.groupby(level=0, group_keys=False).apply(lambda x:(x-x.mean())/x.std())
    label_df = xx2.reset_index()
    
    df_train = pd.merge(df_train, label_df, on = ['end_date','stock_code']).rename(columns={'end_date':'datetime', 'stock_code':'instrument'}).set_index(['datetime','instrument'])
      
    df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    
    stock_index = np.load(args.stock_index_path, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    # slc = slice(args.train_start_date, args.train_end_date)
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train.iloc[:,:-3].astype(np.float32), df_train["label"].astype(np.float32), df_train['market_value'].astype(np.float32), df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    return train_loader #, len(test_time_index)




def main(args):
    seed = np.random.randint(1000000)
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
#     train_loader, valid_loader, test_loader = create_loaders(args)
#     train_loader, valid_loader, test_loader = create_loaders(args)
    # start_dates = ['2012-01-01']
    # start_dates = ['2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01']
    # start_dates=['2014-01-01']
    start_dates=[]
    start_dates.append(args.train_start_date)
    train_loaders = []
    valid_loaders = []
    test_loaders = []
    for start_date in start_dates:
        pprint('start loading', start_date)
        train, valid, test = create_loaders(args, start_date)
        pprint(start_date, ' done...in progress...')
        train_loaders.append(train)
        valid_loaders.append(valid)
        test_loaders.append(test)
    # train_loader, valid_loader, test_loader = create_loaders(args)
    pprint('loaders done')
    out_put_path_base = output_path

    with open(args.stock2concept_matrix_path,'rb') as f:
        stock2concepts=pickle.load(f)
    # if args.model_name=='HIST':
    #     for i in stock2concepts:
    #         stock2concepts[i] = torch.Tensor(stock2concepts[i]).to(device)
        
#     stock2concept_matrix = np.load(args.stock2concept_matrix)
#     if args.model_name == 'HIST':
#         stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    for i in range(0,len(train_loaders)):

        train_loader = train_loaders[i]
        test_loader = test_loaders[i]
        valid_loader = valid_loaders[i]
        output_path = out_put_path_base +'/rolling' + start_dates[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        all_precision = []
        all_recall = []
        all_ic = []
        all_rank_ic = []
        for times in range(args.repeat):
            pprint('create model...')
            if args.model_name == 'SFM':
                model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
            elif args.model_name == 'ALSTM':
                model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
            elif args.model_name == 'Transformer':
                model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
            elif args.model_name == 'HIST':
                model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
            else:
                model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)

            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            best_score = -np.inf
            best_epoch = 0
            stop_round = 0
            best_param = copy.deepcopy(model.state_dict())
            params_list = collections.deque(maxlen=args.smooth_steps)
            for epoch in range(args.n_epochs):
                pprint('Running', times,'Epoch:', epoch)

                pprint('training...')
                train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concepts)
                # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
                # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

                params_ckpt = copy.deepcopy(model.state_dict())
                params_list.append(params_ckpt)
                avg_params = average_params(params_list)
                model.load_state_dict(avg_params)

                pprint('evaluating...')
                train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, stock2concepts, prefix='Train')
                val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, stock2concepts, prefix='Valid')
                pprint('train_ic %.6f, valid_ic %.6f'%(train_ic, val_ic))
                test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, stock2concepts, prefix='Test')

                pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
                pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
                # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
                # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
                pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
                pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
    #                 pprint('Train Precision: ', train_precision)
    #                 pprint('Valid Precision: ', val_precision)
    #                 pprint('Test Precision: ', test_precision)
    #                 pprint('Train Recall: ', train_recall)
    #                 pprint('Valid Recall: ', val_recall)
    #                 pprint('Test Recall: ', test_recall)
                model.load_state_dict(params_ckpt)

                if val_score > best_score:
                    best_score = val_score
                    stop_round = 0
                    best_epoch = epoch
                    best_param = copy.deepcopy(avg_params)
                else:
                    stop_round += 1
                    if stop_round >= args.early_stop:
                        pprint('early stop')
                        break
                pprint('Best epoch: ', best_epoch)
            pprint('best score:', best_score, '@', best_epoch)
            model.load_state_dict(best_param)
            torch.save(best_param, output_path+'/model.bin')

            pprint('inference...')
            res = dict()
            for name in ['test']:

                pred= inference(model, eval(name+'_loader'), stock2concepts)
                pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

                precision, recall, ic, rank_ic = metric_fn(pred)

                pprint(('%s: IC %.6f Rank IC %.6f')%(
                            name, ic.mean(), rank_ic.mean()))
                pprint(name, ': Precision ', precision)
                pprint(name, ': Recall ', recall)
                res[name+'-IC'] = ic
                # res[name+'-ICIR'] = ic.mean() / ic.std()
                res[name+'-RankIC'] = rank_ic
                # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()

            all_precision.append(list(precision.values()))
            all_recall.append(list(recall.values()))
            all_ic.append(ic)
            all_rank_ic.append(rank_ic)

            pprint('save info...')
            writer.add_hparams(
                vars(args),
                {
                    'hparam/'+key: value
                    for key, value in res.items()
                }
            )

            info = dict(
                config=vars(args),
                best_epoch=best_epoch,
                best_score=res,
            )
            default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
            with open(output_path+'/info.json', 'w') as f:
                json.dump(info, f, default=default, indent=4)
        pprint(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
        precision_mean = np.array(all_precision).mean(axis= 0)
        precision_std = np.array(all_precision).std(axis= 0)
        N = [1, 3, 5, 10, 20, 30, 50, 100]
        for k in range(len(N)):
            pprint (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='HIST') # HIST
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=1)

    # data
    parser.add_argument('--data_set', type=str, default='all')
    parser.add_argument('--pin_memory', action='store_false', default=False)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)

    parser.add_argument('--train_start_date', default='2007-01-01') #2007-01-01
    parser.add_argument('--train_end_date', default='2016-12-31') # 2016-12-31
    parser.add_argument('--valid_start_date', default='2017-01-01') # 2017-01-01
    parser.add_argument('--valid_end_date', default='2018-12-31') # 2018-12-31
    parser.add_argument('--test_start_date', default='2019-01-01') # 2019-01-01
    parser.add_argument('--test_end_date', default='2023-10-31') # 2023-10-31
    parser.add_argument('--labels', type=int, default=11)
    parser.add_argument('--begin_days', type=int, default=1)
    parser.add_argument('--feature_list', default=['close','high','low','open','volume','vwap'])
    
    parser.add_argument('--new_start_date', default='2023-01-01')
    parser.add_argument('--new_end_date', default='2023-05-31')
    
    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='csi_HIST')

    # input for csi 
    parser.add_argument('--market_value_path', default='./data2/stock2mkt07_24_date.pkl')
    parser.add_argument('--stock2concept_matrix_path', default='./data2/stock2concept12_6_t_5484.pkl')
    parser.add_argument('--stock_index_path', default='./data2/stock2index._5484.npy')
    parser.add_argument('--feature_path', default='./data2/kbase6/')
    parser.add_argument('--close_path', default='./data2/EODPrices_adjclose.pkl')

    parser.add_argument('--outdir', default='./output/all_HIST_label11to1_CSRankN')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    # main(args)
    
    Train = True

    if Train:
        main(args)
    else:
        loadandinference(args)

