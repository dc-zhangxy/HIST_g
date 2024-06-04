import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.tsa.api as smt

labels = pickle.load(open('HIST_newdata/output51/label1to21.pkl','rb') ).reset_index()
# labels = pickle.load(open('../KRNN/output/label6to21.pkl','rb') ).reset_index()
# labels['datetime'] = labels['datetime'].dt.strftime('%Y-%m-%d')

# dataname = 'all_'+'HIST'+''
# modelname = ['GRU','HIST','HISTdelpre']
# labelname = ['','_label2to1','_label6to1','_label11to1']

# datanames = ['all_ind1' + '_label11to1_' + 'HGCN',
#             'all_ind2' + '_label11to1_' + 'HGCN']
# datanames = ['all_label11to1_HGCN1_amend_analyst',
#             'all_label11to1_HGCN1_amend_prod',
#             'all_flow_label11to1_HGCN1_flow']
# labels['instrument'] =  labels['instrument'].apply(lambda x: str(x)[2:])

# datanames = ['all_label11to1_HGAT_09']
# datanames = ['all_label21to1_GRU',
#             'all_label21to1_HISTdelpre',
#             'allind1_label21_GCNind1_0930',
#             'all_label21to1_GRU_icloss',
#             'all_label21to1_HISTdelpre_icloss',
#             'allind1_label21_GCN_icloss_1130',]
# datanames = ['comb',
#             'comb21',
#             'combGHC21',
#             'combHC21']
datanames = ['Alpha360_label11']

# for b in labelname:
#     for a in modelname:
        # dataname = 'all_'+a+b
        # dir_ = 'HIST_newdata/output51/'+dataname+'/rolling2009-01-01/pred.pkl.test0'
for d in datanames:
    # dataname = 'all_' + s + '_label11to1_' + modelname
    # dataname = 'all_' +'label11to1_' + s 
    # dir_ = 'HIST_gcn/output51gcn/'+d+'/rolling2009-01-01/pred.pkl.test0'
    # dir_ = 'HIST_gcn/output53gcn/'+d+'/rolling2005-01-01/pred.pkl.test0'
    # dir_ = d+'.pkl'
    dir_ = '../TRA/output51/'+d+'/test_pred.pkl'
    GRU_1 = pickle.load(open(dir_ ,'rb'))
    # GRU_1.columns = ['score1']
    GRU_1 = GRU_1.reset_index()
    
    if type(GRU_1['datetime'][0]) == str:
        GRU_1['datetime'] = [datetime.strptime(a, '%Y-%m-%d') for a in GRU_1['datetime']]
    
    pred_all = pd.merge(GRU_1, labels) # .drop('label1',axis=1)
    pred_all1 = pred_all.set_index(['datetime','instrument'])
    # pred_all_norm = pred_all1.groupby(level='datetime', group_keys=False).apply(lambda x:(x-x.mean())/x.std())
    # pred_all_norm.corr(method='spearman')*100
    with open('acf.txt','a',encoding='utf-8') as f:
        for i in [6,11,21]:
            rankic = pred_all1.groupby(level='datetime', group_keys=False).apply(lambda x: x['label'+str(i)].corr(x['score'], method='spearman')).mean()
            f.write(str(rankic*100)+'\t')
        f.write('\n')
    # with open('acf.txt','a',encoding='utf-8') as f:
    #     f.write('\n')

