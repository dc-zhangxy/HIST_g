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
# import qlib
from torch.utils.tensorboard import SummaryWriter
# from qlib.contrib.model.pytorch_gru import GRUModel
# from qlib.contrib.model.pytorch_transformer import Transformer
from pytorch_transformer import Transformer
from utils import metric_fn, mse
from dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    # if model_name.upper() == 'GRU':
    #     return GRUModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

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
    # mask = ~torch.isnan(label)
    return mse(pred, label) #mse(pred[mask], label[mask])


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
def train_epoch(epoch, model, optimizer, train_loader, writer, args):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, index = train_loader.get(slc)

        pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, index = test_loader.get(slc)

        with torch.no_grad():

            pred = model(feature)
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)

    scores = -1.0 * np.mean(losses)

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)


    return np.mean(losses), scores #, precision, recall, ic, rank_ic

def inference(model, data_loader):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, index = data_loader.get(slc)

        with torch.no_grad():

            pred = model(feature)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def loadandinference(args):
    pprint('create model...')
    if args.model_name == 'TRANSFORMER':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=args.dropout)
    else:
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)

    model.to(device)
    output_path = args.outdir

    best_param = output_path +'/rolling' + '2007-01-01' + '/model.bin'
    model.load_state_dict(torch.load(best_param))
    # model.eval()

    pprint('loading data...')

    newdataloader = create_newloader(args)
    # name = 'newdata'

    pprint('inference...')

    pred= inference(model, newdataloader)
    pred.to_pickle(output_path+'/newdata_pred.pkl')

    # precision, recall, ic, rank_ic = metric_fn(pred)
    # pprint(('%s: IC %.6f Rank IC %.6f')%(
    #             name, ic.mean(), rank_ic.mean()))
    # pprint(name, ': Precision ', precision)
    # pprint(name, ': Recall ', recall)
    pprint('finished.')
    
def create_newloader(args):
    Feature = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'VALUE']
    Columns = []
    for f in Feature:
        for i in range(args.days-1,0,-1):
            Columns.append(f+'_lag_'+str(i))
        Columns.append(f)
        
    dev_test_mul = pd.read_csv('dev_test_mul.csv')
    dev_test_mul = dev_test_mul.set_index(['datetime', 'instrument']).sort_index(level=['datetime', 'instrument'])
    test_loader = DataLoader(dev_test_mul[Columns], dev_test_mul["label"], pin_memory=False, start_index=0, device = device)

    return test_loader

def create_loaders(args,train_start_date):

    Feature = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'VALUE']
    Columns = []
    for f in Feature:
        for i in range(args.days-1,0,-1):
            Columns.append(f+'_lag_'+str(i))
        Columns.append(f)
        
    dev_train_mul = pd.read_csv('dev_train_mul'+str(args.labels)+'_'+str(args.days)+'.csv')
    # 用于 指数集 的训练
    # dev_train_mul = dev_train_mul[dev_train_mul['instrument'].isin([1,16,300,812,813,906,922,949,861,859])]
    # 用于 单个指数 的训练
    # dev_train_mul = dev_train_mul[dev_train_mul['instrument']==1]
    
    train_time = dev_train_mul['datetime'].drop_duplicates().to_list()
    dev_test_mul = pd.read_csv('dev_test_mul'+str(args.labels)+'_'+str(args.days)+'.csv')
    dev_train_mul = dev_train_mul.set_index(['datetime', 'instrument']).sort_index(level=['datetime', 'instrument'])
    dev_test_mul = dev_test_mul.set_index(['datetime', 'instrument']).sort_index(level=['datetime', 'instrument'])
    
    dev_train = dev_train_mul.loc[train_time[:int(0.9*len(train_time))]]
    dev_valid = dev_train_mul.loc[train_time[int(0.9*len(train_time)):]]
    # df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)

    start_index = 0

    train_loader = DataLoader(dev_train[Columns], dev_train["label"], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    start_index += len(dev_test_mul.groupby(level=0).size())

    valid_loader = DataLoader(dev_valid[Columns], dev_valid["label"],  pin_memory=False, start_index=start_index, device = device)
    
    start_index += len(dev_test_mul.groupby(level=0).size())

    test_loader = DataLoader(dev_test_mul[Columns], dev_test_mul["label"], pin_memory=False, start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader


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

    for i in range(0,len(train_loaders)):

        train_loader = train_loaders[i]
        test_loader = test_loaders[i]
        valid_loader = valid_loaders[i]
        output_path = out_put_path_base +'/rolling' + start_dates[i]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # all_precision = []
        # all_recall = []
        # all_ic = []
        # all_rank_ic = []
        for times in range(args.repeat):
            pprint('create model...')
            if args.model_name == 'TRANSFORMER':
                model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=args.dropout)
            else:
                model = get_model(args.model_name)()

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
                train_epoch(epoch, model, optimizer, train_loader, writer, args)
                # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
                # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

                params_ckpt = copy.deepcopy(model.state_dict())
                params_list.append(params_ckpt)
                avg_params = average_params(params_list)
                model.load_state_dict(avg_params)

                pprint('evaluating...')
                # train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
                # val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
                train_loss, train_score = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
                val_loss, val_score = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
                
                # pprint('train_ic %.6f, valid_ic %.6f'%(train_ic, val_ic))
                # test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')
                test_loss, test_score = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')

                pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
                pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))

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

                pred= inference(model, eval(name+'_loader'))
                pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

                # precision, recall, ic, rank_ic = metric_fn(pred)

                # pprint(('%s: IC %.6f Rank IC %.6f')%(
                #             name, ic.mean(), rank_ic.mean()))
                # pprint(name, ': Precision ', precision)
                # pprint(name, ': Recall ', recall)
                # res[name+'-IC'] = ic
                # res[name+'-RankIC'] = rank_ic

            # all_precision.append(list(precision.values()))
            # all_recall.append(list(recall.values()))
            # all_ic.append(ic)
            # all_rank_ic.append(rank_ic)

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
        # pprint(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
        # precision_mean = np.array(all_precision).mean(axis= 0)
        # precision_std = np.array(all_precision).std(axis= 0)
        # N = [1, 3, 5, 10, 20, 30, 50, 100]
        # for k in range(len(N)):
        #     pprint (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def data_prepare(filepath, mode, args):
    # import pandas as pd
    test = pd.read_csv(filepath, sep = ',')

    stocks = test['STOCK_CODE'].drop_duplicates().to_list()
    df1 = pd.DataFrame()
    # 按股票名分别插入对应 NEXT_CLOSE
    for i in range(len(stocks)):
        df2 = test[test['STOCK_CODE']==stocks[i]].reset_index(drop=True)
        # 预测几天标签
        df2['NEXT_CLOSE'] = df2['CLOSE'][args.labels:].reset_index(drop=True)
        df1 = pd.concat([df1, df2])
        
    test1 = df1.dropna().reset_index(drop=True).drop('Unnamed: 0',axis=1)
    # .to_csv('test1.csv',index=False)

    # 所有特征列，除了'STOCK_CODE'和'END_DATE'
    feature_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE','VOL', 'VALUE']

    # 创建一个空的DataFrame来存储结果
    expanded_df = pd.DataFrame()

    # 对每一个股票代码处理
    for code in test1['STOCK_CODE'].unique():
        # 选取当前股票代码的数据
        stock_data = test1[test1['STOCK_CODE'] == code].copy()

        # 对每一个特征列生成过去60天的数据
        for column in feature_columns:
            for i in range(1, args.days+1):  # 从1到60天
                stock_data[f'{column}_lag_{i}'] = stock_data[column].shift(i)

        # 添加到结果DataFrame
        expanded_df = pd.concat([expanded_df, stock_data], axis=0)

    # 删除含有NaN的行（因为这些行没有足够的历史数据）
    expanded_df.dropna(inplace=True)
    # 创建1日标签
    expanded_df['label'] = expanded_df['NEXT_CLOSE'].div( expanded_df['CLOSE'], axis=0)-1

    Feature = ['OPEN', 'HIGH', 'LOW', 'CLOSE'] # , ['VOL', 'VALUE']
    Columns_c = ['OPEN', 'HIGH', 'LOW', 'CLOSE'] # 需要除以close的特征
    for i in range(1,args.days+1):
        for f in Feature:
            Columns_c.append(f+'_lag_'+str(i))
    expanded_df[Columns_c] = expanded_df[Columns_c].div( expanded_df['CLOSE'], axis=0)-1

    # Feature = ['VOL', 'VALUE'] # 需要除以vol和value以归一化的特征
    Columns_vol = ['VOL']
    Columns_value = ['VALUE']
    for i in range(1,args.days+1):
        Columns_vol.append('VOL'+'_lag_'+str(i))
        Columns_value.append('VALUE'+'_lag_'+str(i))
            
    expanded_df[Columns_vol] = expanded_df[Columns_vol].div( expanded_df['VOL'], axis=0) -1
    expanded_df[Columns_value] = expanded_df[Columns_value].div( expanded_df['VALUE'], axis=0) -1

    dev_test = expanded_df.drop(['PRE_CLOSE','NEXT_CLOSE'],axis=1)
    #.to_csv('dev_test.csv', index=False)
    dev_test  = dev_test.rename(columns={"STOCK_CODE": "instrument", "END_DATE": "datetime"})
    dev_test = dev_test.set_index(['datetime', 'instrument']).sort_index(level=['datetime', 'instrument'])
    # dev_test
    dev_test.to_csv('dev_'+mode+'_mul'+str(args.labels)+'_'+str(args.days)+'.csv')


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='TRANSFORMER')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    # parser.add_argument('--metric', default='loss')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--filepath', type=str, default='./train.csv')

    # data
    parser.add_argument('--data_set', type=str, default='all')
    parser.add_argument('--pin_memory', action='store_false', default=False)
    parser.add_argument('--batch_size', type=int, default=1000) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    
    parser.add_argument('--train_start_date', default='2007-01-01') #2009-01-01
    parser.add_argument('--train_end_date', default='2017-10-30') # 2016-12-31
    parser.add_argument('--valid_start_date', default='2017-10-31') # 2017-01-01
    parser.add_argument('--valid_end_date', default='2018-12-31') # 2018-12-31
    parser.add_argument('--test_start_date', default='2019-01-01') # 2019-01-01
    parser.add_argument('--test_end_date', default='2022-12-31') # 2022-12-31
    parser.add_argument('--labels', type=int, default=1)
    parser.add_argument('--days', type=int, default=90) # specify other labels

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='all_transf')

    parser.add_argument('--outdir', default='./output/all_transf_loss_valid_all_value_label5_day90')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    
    # 数据预处理
    data_prepare('../train.txt','train',args)
    data_prepare('../test.txt','test',args)

    if args.mode == 'train':
        main(args)
    elif args.mode == 'test':
        loadandinference(args)
    else:
        pprint('mode name error!')
