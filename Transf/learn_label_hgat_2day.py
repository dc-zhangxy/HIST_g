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
import qlib
# regiodatetimeG_CN, REG_US]
# from dateutil.relativedelta import relativedelta
from qlib.config import REG_US, REG_CN
# from dateutil.relativedelta import relativedelta
# provider_uri = "../HIST_all/data/cn_data_updated"  # target_dir
provider_uri = "../datafolder/cn_data_updated"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
# from qlib.contrib.model.pytorch_lstm import LSTMModel
# from qlib.contrib.model.pytorch_gats import GATModel
# from qlib.contrib.model.pytorch_sfm import SFM_Model
# from qlib.contrib.model.pytorch_alstm import ALSTMModel
# from qlib.contrib.model.pytorch_transformer import Transformer
# from pytorch_transformer import Transformer
# from hist_delpre_model import MLP, HIST   #model/ model2
# from gcn_models import GCN
# from hist_gcn_model import HIST_GCN
# from hist_gat_model import HIST_GAT
from utils_2day import metric_fn, mse, cal_cos_similarity, cal_convariance
from dataloader_2day import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    # if model_name.upper() == 'MLP':
    #     return MLP

    # if model_name.upper() == 'LSTM':
    #     return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    # if model_name.upper() == 'GATS':
    #     return GATModel

    # if model_name.upper() == 'SFM':
    #     return SFM_Model

    # if model_name.upper() == 'ALSTM':
    #     return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        from pytorch_transformer import Transformer
        return Transformer

    if model_name.upper() == 'FULL_TRANSF':
        return torch.nn.Transformer
        # from full_transf import Transformer
        # return Transformer
    
    # if model_name.upper() == 'GCN':
    #     return HIST_GCN #GCN
    
    # if model_name.upper() == 'HGAT':
    #     return HIST_GAT #HIST_GCN

    raise ValueError('unknown model name `%s`'%model_name)

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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
def train_epoch(epoch, model, optimizer, train_loader, writer, args):

    global global_step

    model.train()

    for i, slc, i2, slc2 in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
   
        feature, label, index = train_loader.get(slc)
        pred = model(feature)
        scores1 = pd.DataFrame( pred.detach().cpu().numpy(), columns = ['day1score'], index=index)
        # print(pred.shape, label.shape) # torch.Size([285]) torch.Size([285])
        
        feature2, label2, index2 = train_loader.get(slc2)
        pred2 = model(feature2)
        scores2 = pd.DataFrame( pred2.detach().cpu().numpy(), columns = ['day2score'], index=index2)
        
        scores12 = pd.concat([scores1, scores2], axis=1, join='inner')
        # corr = scores12.groupby(level='datetime').apply(lambda x: x['day1score'].corr(x['day2score'], method='spearman')).mean()
        corr = scores12['day1score'].corr(scores12['day2score']) 
        
        loss = loss_fn(pred, label, args) + loss_fn(pred2, label2, args) - 1*corr #torch.cosine_similarity(pred,pred2,dim=0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
        
 # train 用真实值train，但test用预测值test       
def test(model, enc_input, start_symbol=torch.zeros(5)):
    tgt_len = 5
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    # enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model(enc_input, dec_input)
        # projected = model.projection(dec_outputs)
        # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # next_word = prob.data[i]
        next_symbol = dec_outputs[i]
    return dec_input

def test_epoch(epoch, model, test_loader, writer, args, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, index = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'FULL_TRANSF':
                # src [N, F*T] --> [N, T, F]  [Batch_size, 60, 6]
                ba = len(label)
                enc_inputs = feature.reshape(ba, 6, -1).permute(0, 2, 1)
                label_re = label.reshape(ba, -1, 6)
                # da = label_re.shape[1]
                
                dec_inputs = torch.cat([torch.zeros([ba,1,6], device = device), label_re], dim=1)   # Decoder_input 首位为起始符号0 (1
                # dec_outputs = torch.cat([label_re,torch.ones(ba,1,6)], dim=1) # Decoder_output 末位为结束符号1 (2
                pred = model(enc_inputs, dec_inputs, tgt_mask=generate_square_subsequent_mask(5).to(device))[:,:-1,:]
                loss = loss_fn(pred, label_re, args)
                scoress = pd.DataFrame(pred[:,:,0].reshape(ba,-1).cpu().numpy(), columns = ['score1','score2','score3','score4'], index=index)
                labelss = pd.DataFrame(label_re[:,:,0].reshape(ba,-1).cpu().numpy(), columns = ['label1','label2','label3','label4'], index=index)
            else:
                pred = model(feature)
                loss = loss_fn(pred, label, args)
                scoress = pd.DataFrame( pred.cpu().numpy(), columns = ['score'], index=index)
                labelss = pd.DataFrame( label.cpu().numpy(), columns = ['label'], index=index)
            
            preds.append(pd.concat([scoress,labelss],axis=1))
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

def inference(model, data_loader):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, index = data_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'FULL_TRANSF':
                # src [N, F*T] --> [N, T, F]  [Batch_size, 60, 6]
                ba = len(label)
                enc_inputs = feature.reshape(ba, 6, -1).permute(0, 2, 1)
                label_re = label.reshape(ba, -1, 6)
                # da = label_re.shape[1]
                
                dec_inputs = torch.cat([torch.zeros([ba,1,6], device = device), label_re], dim=1)   # Decoder_input 首位为起始符号0 (1
                # dec_outputs = torch.cat([label_re,torch.ones(ba,1,6)], dim=1) # Decoder_output 末位为结束符号1 (2
                pred = model(enc_inputs, dec_inputs, tgt_mask=generate_square_subsequent_mask(5).to(device))[:,:-1,:]
                loss = loss_fn(pred, label_re, args)
                scoress = pd.DataFrame(pred[:,:,0].reshape(ba,-1).cpu().numpy(), columns = ['score1','score2','score3','score4'], index=index)
                labelss = pd.DataFrame(label_re[:,:,0].reshape(ba,-1).cpu().numpy(), columns = ['label1','label2','label3','label4'], index=index)
            else:
                pred = model(feature)
                loss = loss_fn(pred, label, args)
                scoress = pd.DataFrame( pred.cpu().numpy(), columns = ['score'], index=index)
                labelss = pd.DataFrame( label.cpu().numpy(), columns = ['label'], index=index)
            
            preds.append(pd.concat([scoress,labelss],axis=1))
            # preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))
    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args,train_start_date):
    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')
    hanlder = {'class': 'Alpha360', 'module_path': 'handler.py', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                'label': [f'Ref($close,-21) / Ref($close,-1) - 1']
        #         'label': ['Ref($close,-6) / Ref($close,-1) - 1','Ref($open,-6)','Ref($high,-6)','Ref($low,-6)','Ref($vwap,-6)','Ref($volume,-6)',
        #   'Ref($close,-11) / Ref($close,-1) - 1','Ref($open,-11)','Ref($high,-11)','Ref($low,-11)','Ref($vwap,-11)','Ref($volume,-11)',
        #   'Ref($close,-16) / Ref($close,-1) - 1','Ref($open,-16)','Ref($high,-16)','Ref($low,-16)','Ref($vwap,-16)','Ref($volume,-16)',
        #   'Ref($close,-21) / Ref($close,-1) - 1','Ref($open,-21)','Ref($high,-21)','Ref($low,-21)','Ref($vwap,-21)','Ref($volume,-21)']     
                }}
    segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder,segments)
    print(f'train: ({args.train_start_date}, {args.train_end_date}), valid: ({args.valid_start_date}, {args.valid_end_date}), test: ({args.test_start_date}, {args.test_end_date})')
    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)

    start_index = 0

    train_loader = DataLoader(df_train["feature"], df_train["label"], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"],  pin_memory=True, start_index=start_index, device = device)
    
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], pin_memory=True, start_index=start_index, device = device)

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
    global_log_file = output_path + '/' +'run.log'

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
        all_precision = []
        all_recall = []
        all_ic = []
        all_rank_ic = []
        for times in range(args.repeat):
            pprint('create model...')
            if args.model_name == 'SFM':
                model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
            # elif args.model_name == 'ALSTM':
            #     model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
            elif args.model_name == 'Transformer':
                model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.1) # 0.5
            elif args.model_name == 'FULL_TRANSF':
                model = get_model(args.model_name)(d_model=6, nhead=6, num_encoder_layers=6, num_decoder_layers=6, batch_first=True)
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
                torch.save(model.state_dict(), output_path+'/modele.bin')
                torch.save(optimizer.state_dict(), output_path+'/optimizere.bin')

                params_ckpt = copy.deepcopy(model.state_dict())
                params_list.append(params_ckpt)
                avg_params = average_params(params_list)
                model.load_state_dict(avg_params)

                pprint('evaluating...')
                train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
                val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
                pprint('train_ic %.6f, valid_ic %.6f'%(np.mean(train_ic), np.mean(val_ic)))
                test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')

                pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(np.mean(train_loss), np.mean(val_loss), np.mean(test_loss)))
                pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(np.mean(train_score), np.mean(val_score), np.mean(test_score)))

                pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(np.mean(train_ic), np.mean(val_ic), np.mean(test_ic)))
                pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(np.mean(train_rank_ic), np.mean(val_rank_ic), np.mean(test_rank_ic)))
                # pprint('train_ic %.6f'%(np.mean(train_ic)))
    #                 pprint('Valid Precision: ', val_precision)
                    # pprint('Test Precision: ', test_precision)
                # pprint('train_loss %.6f'%(np.mean(train_loss)))
    #                 pprint('Valid Recall: ', val_recall)
    #                 pprint('Test Recall: ', test_recall)
                model.load_state_dict(params_ckpt)

                if np.mean(val_score) > best_score:
                    best_score = np.mean(val_score)
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
            for name in ['train', 'valid', 'test']:

                pred= inference(model, eval(name+'_loader'))
                pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

                precision, recall, ic, rank_ic = metric_fn(pred)

                pprint(('%s: IC %.6f Rank IC %.6f')%(
                            name, np.mean(ic), np.mean(rank_ic)))
                pprint(name, ': Precision ', precision)
                pprint(name, ': Recall ', recall)
                res[name+'-IC'] = np.mean(ic)
                # res[name+'-ICIR'] = ic.mean() / ic.std()
                res[name+'-RankIC'] = np.mean(rank_ic)
                # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()

            all_precision.append(list(i.values() for i in precision))
            all_recall.append(list(i.values() for i in recall))
            all_ic.append(np.mean(ic))
            all_rank_ic.append(np.mean(rank_ic))

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


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='GRU')  # TRANSFORMER  FULL_TRANSF
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=1)

    # data
    parser.add_argument('--data_set', type=str, default='all')
    parser.add_argument('--pin_memory', action='store_false', default=False)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2005-01-01') #2009-01-01
    parser.add_argument('--train_end_date', default='2016-09-30') # 2016-12-31
    parser.add_argument('--valid_start_date', default='2017-01-01') # 2017-01-01
    parser.add_argument('--valid_end_date', default='2018-09-30') # 2018-12-31
    parser.add_argument('--test_start_date', default='2019-01-01') # 2019-01-01
    parser.add_argument('--test_end_date', default='2023-12-31') # 2022-12-31
    parser.add_argument('--labels', type=int, default=2)

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    # parser.add_argument('--name', type=str, default='csi300_HIST')

    # input for csi 300
    # parser.add_argument('--market_value_path', default='./data_2/stock2mkt.pkl')
    # parser.add_argument('--stock2concept_matrix', default='./data_2/stock2concept.pkl')
    # parser.add_argument('--stock_index', default='./data_2/stock2index.npy')

    parser.add_argument('--outdir', default='./output/all360_GRU_2day_corr1_label20')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
