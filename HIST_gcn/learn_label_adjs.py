import torch
import torch.nn as nn
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
import qlib
# regiodatetimeG_CN, REG_US]
from dateutil.relativedelta import relativedelta
from qlib.config import REG_US, REG_CN
# from dateutil.relativedelta import relativedelta
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# provider_uri = "../data/cn_data_updated"  # target_dir
provider_uri = "../datafolder/cn_data_updated"  # target_dir
# provider_uri = "/home/zhangxinyi/HIST_g/datafolder/cn_data_updated"
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gats import GATModel
from qlib.contrib.model.pytorch_sfm import SFM_Model
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.model.pytorch_transformer import Transformer
from hist_delpre_model import MLP, HIST   #model/ model2
from gcn_models import GCN
from hist_gcn_model import HIST_GCN
from hist_gat_model import HIST_GAT
from utils import metric_fn, mse, ic_loss
# from dataloader_analyst import DataLoader
# from dataloader_flow import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12

def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    if model_name.upper() == 'GATS':
        return GATModel

    if model_name.upper() == 'SFM':
        return SFM_Model

    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'HIST':
        return HIST
    
    if model_name.upper() == 'GCN':
        return HIST_GCN #GCN
    if model_name.upper() == 'HGAT':
        return HIST_GAT #HIST_GCN

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
    return ic_loss(pred[mask], label[mask]) # mse


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
def train_epoch(epoch, model, optimizer, train_loader, train_end_index, writer, args):

    global global_step
    model.train()

    for slc in range(train_end_index):
        global_step += 1
        feature, label, adj , stock_index, day_index = train_loader.get(slc)

        if args.model_name == 'GCN':
            pred = model(feature, adj)
        else:
            pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader,test_end_index, writer, args, prefix='Test'):

    model.eval()
    losses = []
    preds = []

    for slc in range(test_end_index):
        feature, label, adj , stock_today, day_index = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'GCN':
                pred = model(feature, adj)
            else:
                pred = model(feature)
            loss = loss_fn(pred, label, args)

            preds_df = pd.DataFrame({ 'score': pred.cpu().detach().numpy(), 'label': label.cpu().detach().numpy(), 'datetime':[day_index for i in range(len(stock_today))] ,'instrument':stock_today})
            preds.append(preds_df.set_index(['datetime','instrument'],drop=True))
            # preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

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

def inference(model, data_loader, end_index):

    model.eval()
    preds = []

    for slc in range(end_index):
        feature, label, adj , stock_today, day_index = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'GCN':
                pred = model(feature, adj)
            else:
                pred = model(feature)

            preds_df = pd.DataFrame({ 'score': pred.cpu().detach().numpy(), 'label': label.cpu().detach().numpy(), 'datetime':[day_index for i in range(len(stock_today))] ,'instrument':stock_today})
            preds.append(preds_df.set_index(['datetime','instrument'],drop=True))
            # preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args, train_start_date):

    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 
            'instruments': args.data_set, 
            'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 
            'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
            'label': [f'Ref($close,{-args.labels}) / Ref($close,-1) - 1']}}
    segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder,segments)
    print(f'train: ({args.train_start_date}, {args.train_end_date}), valid: ({args.valid_start_date}, {args.valid_end_date}), test: ({args.test_start_date}, {args.test_end_date})')
    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)

    train_time = df_train.reset_index()['datetime'].drop_duplicates().to_list()
    train_time_index = [str(t)[:10] for t in train_time]
    valid_time = df_valid.reset_index()['datetime'].drop_duplicates().to_list()
    valid_time_index = [str(t)[:10] for t in valid_time]
    test_time = df_test.reset_index()['datetime'].drop_duplicates().to_list()
    test_time_index = [str(t)[:10] for t in test_time]
    # stock_index = np.load(args.stock_index, allow_pickle=True).item()

    if args.adj_name == 'industry':
        # data_set='csi800'
        data_dir = 'data/'

        csi800 = pd.read_csv(provider_uri+'/instruments/csiall_industry.txt', sep='\t',header=None)
        csi800list = csi800[0].drop_duplicates().to_list() #  1804

        industry = pd.read_excel(data_dir+'industry.xlsm')  # 行业数据
        industry['stock_code2'] = [a[-2:]+a[:-3] for a in industry['stock_code']]

        industry800 = industry[industry['stock_code2'].isin(csi800list)].reset_index(drop=True)

        stock800list = csi800list 
        len800 = len(stock800list)
        # stock_dict = dict(zip(stock800list,range(len800)))

        np_adj = np.zeros([len800,len800])
        for i in range(len800):
            for j in range(len800):
                if industry800['行业1'][i] == industry800['行业1'][j]:
                    np_adj[i,j] = 1.0

        from dataloader_industry2 import DataLoader
        train_loader = DataLoader(df_train["feature"], df_train["label"], np_adj, csi800list, train_time_index, device = device)
        valid_loader = DataLoader(df_valid["feature"], df_valid["label"], np_adj, csi800list, valid_time_index, device = device)
        test_loader = DataLoader(df_test["feature"], df_test["label"], np_adj, csi800list, test_time_index, device = device)

    else:
        if args.adj_name == 'analyst':
            dir_ = 'data/adjacency_analyst.pkl' 
            from dataloader_analyst import DataLoader
        elif args.adj_name == 'product':
            dir_ = 'data/adj_myprod.pkl'
            from dataloader_analyst import DataLoader
        elif args.adj_name == 'inflow_trade':
            dir_ = 'data/inflow_trade.pkl'
            from dataloader_flow_2 import DataLoader

        with open(dir_,'rb')as f: # 
            adj_analyst = pickle.load(f)
        
        train_loader = DataLoader(df_train["feature"], df_train["label"], adj_analyst, train_time_index, device = device)
        valid_loader = DataLoader(df_valid["feature"], df_valid["label"], adj_analyst, valid_time_index, device = device)
        test_loader = DataLoader(df_test["feature"], df_test["label"], adj_analyst, test_time_index, device = device)

    return train_loader, valid_loader, test_loader, len(train_time_index), len(valid_time_index), len(test_time_index)


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
    train_lens = []
    valid_lens = []
    test_lens = []
    for start_date in start_dates:
        pprint('start loading', start_date)
        train, valid, test, train_len, valid_len, test_len = create_loaders(args, start_date)
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
            elif args.model_name == 'GCN':
                model = get_model(args.model_name)()
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
                train_epoch(epoch, model, optimizer, train_loader, train_ilens, writer, args)
                # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
                # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

                params_ckpt = copy.deepcopy(model.state_dict())
                params_list.append(params_ckpt)
                avg_params = average_params(params_list)
                model.load_state_dict(avg_params)

                pprint('evaluating...')
                train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, train_ilens, writer, args, prefix='Train')
                val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, valid_ilens, writer, args, prefix='Valid')
                pprint('train_ic %.6f, valid_ic %.6f'%(train_ic, val_ic))
                test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, test_ilens, writer, args, prefix='Test')

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

                pred= inference(model, eval(name+'_loader'), eval(name+'_ilens'))
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

    model.to(device)
    output_path = args.outdir

    best_param = output_path +'/rolling' + '2009-01-01' + '/model.bin'
    model.load_state_dict(torch.load(best_param))
    # model.eval()

    pprint('loading data...')

    newdataloader, new_ilens = create_newloader(args)
    name = 'newdata'

    pprint('inference...')

    pred= inference(model, newdataloader, new_ilens)
    pred.to_pickle(output_path+'/newdata_pred.pkl')

    precision, recall, ic, rank_ic = metric_fn(pred)

    pprint(('%s: IC %.6f Rank IC %.6f')%(
                name, ic.mean(), rank_ic.mean()))
    pprint(name, ': Precision ', precision)
    pprint(name, ': Recall ', recall)


def create_newloader(args):

    start_time = datetime.datetime.strptime(args.new_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.new_end_date, '%Y-%m-%d')
    # train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': start_time, 
            'instruments': args.data_set, 
            'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 
            'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
            'label': [f'Ref($close,{-args.labels}) / Ref($close,-1) - 1']}}
    
    segments =  {'test': (args.new_start_date, args.new_end_date)}
    dataset = DatasetH(hanlder,segments)

    print(f'test: ({args.new_start_date}, {args.new_end_date})')
    df_test = dataset.prepare( "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    
    test_time = df_test.reset_index()['datetime'].drop_duplicates().to_list()
    test_time_index = [str(t)[:10] for t in test_time]
    # stock_index = np.load(args.stock_index, allow_pickle=True).item()

    if args.adj_name == 'industry':
        # data_set='csi800'
        data_dir = 'data/'

        csi800 = pd.read_csv(provider_uri+'/instruments/csiall_industry.txt', sep='\t',header=None)
        csi800list = csi800[0].drop_duplicates().to_list() #  1804

        industry = pd.read_excel(data_dir+'industry.xlsm')  # 行业数据
        industry['stock_code2'] = [a[-2:]+a[:-3] for a in industry['stock_code']]

        industry800 = industry[industry['stock_code2'].isin(csi800list)].reset_index(drop=True)

        stock800list = csi800list 
        len800 = len(stock800list)
        # stock_dict = dict(zip(stock800list,range(len800)))

        np_adj = np.zeros([len800,len800])
        for i in range(len800):
            for j in range(len800):
                if industry800['行业1'][i] == industry800['行业1'][j]:
                    np_adj[i,j] = 1.0

        from dataloader_industry2 import DataLoader
        
        test_loader = DataLoader(df_test["feature"], df_test["label"], np_adj, csi800list, test_time_index, device = device)

    else:
        if args.adj_name == 'analyst':
            dir_ = 'data/adjacency_analyst.pkl' 
            from dataloader_analyst import DataLoader
        elif args.adj_name == 'product':
            dir_ = 'data/adj_myprod.pkl'
            from dataloader_analyst import DataLoader
        elif args.adj_name == 'inflow_trade':
            dir_ = 'data/inflow_trade.pkl'
            from dataloader_flow import DataLoader

        with open(dir_,'rb')as f: # 
            adj_analyst = pickle.load(f)
        
        test_loader = DataLoader(df_test["feature"], df_test["label"], adj_analyst, test_time_index, device = device)

    return test_loader, len(test_time_index)



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
    parser.add_argument('--model_name', default='GCN')
    parser.add_argument('--adj_name', default='industry')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=3)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=1)

    # data
    parser.add_argument('--data_set', type=str, default='all_industry')
    parser.add_argument('--pin_memory', action='store_false', default=False)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2005-01-01') # 2009-01-01  2005-01-01
    parser.add_argument('--train_end_date', default='2016-11-30')
    parser.add_argument('--valid_start_date', default='2017-01-01')
    parser.add_argument('--valid_end_date', default='2018-11-30')
    parser.add_argument('--test_start_date', default='2019-01-01')
    parser.add_argument('--test_end_date', default='2023-12-31')
    parser.add_argument('--new_start_date', default='2023-01-01')
    parser.add_argument('--new_end_date', default='2023-05-31')
    parser.add_argument('--labels', type=int, default=21)

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='csi300_HIST')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data_2/stock2mkt.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data_2/stock2concept.pkl')
    parser.add_argument('--stock_index', default='./data_2/stock2index.npy')

    parser.add_argument('--outdir', default='./output/all_label21_GCNind1_icloss')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    Train = True

    if Train:
        main(args)
    else:
        loadandinference(args)

