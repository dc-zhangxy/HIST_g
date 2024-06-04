import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.tsa.api as smt

# labels = pickle.load (open('HIST_newdata/output51/label1to21.pkl','rb') ).reset_index()

datanum = 'csi800_'
labelt = 'label21to1_'
# with open('HIST_newdata/output6/'+datanum+'GRU'+labelt+'/rolling2009-01-01/pred.pkl.test0','rb')as f: #
with open('HIST_gcn/output51gcn/'+datanum+labelt+'GRU_90'+'/rolling2005-01-01/pred.pkl.test0','rb')as f: # 
# with open('../KRNN/output/'+datanum+labelt+'KRNN/rolling2005-01-01/pred.pkl.test0','rb')as f: # 
    pred3=pickle.load(f)
pred3.columns = ['score300','label300']
pred3 = pred3.reset_index()


# dataname = 'all_'+'HIST'+''
# modelname = ['GRU','HIST','HISTdelpre']
# labelname = ['','_label2to1','_label6to1','_label11to1']
# datanames = ['all_label11to1_HGAT_09']
# datanames = ['all_ind1' + '_label11to1_' + 'HGCN',
#             'all_ind2' + '_label11to1_' + 'HGCN']
# datanames = ['all_label11to1_HGCN1_amend_analyst',
#             'all_label11to1_HGCN1_amend_prod',
#             'all_flow_label11to1_HGCN1_flow']
# pred3['instrument'] =  pred3['instrument'].apply(lambda x: str(x)[2:])

# datanames = ['all_label21to1_GRU',
#             'all_label21to1_HISTdelpre',
#             'allind1_label21_GCNind1_0930',
#             'all_label21to1_GRU_icloss',
#             'all_label21to1_HISTdelpre_icloss',
#             'allind1_label21_GCN_icloss_1130',]
datanames = ['comb',
            'comb21',
            'combGHC21',
            'combHC21']

# for b in labelname:
#     for a in modelname:
for d in datanames:
        # dataname = 'all_'+a+b
        # dir_ = 'HIST_newdata/output51/'+dataname+'/rolling2009-01-01/pred.pkl.test0'
        # dir_ = 'HIST_gcn/output/'+d+'/rolling2009-01-01/pred.pkl.test0'
        # dir_ = 'HIST_gcn/output53gcn/'+d+'/rolling2005-01-01/pred.pkl.test0'
        dir_ = d+'.pkl'
        GRU_1=pickle.load(open(dir_,'rb'))
        GRU_1.columns = ['score1'] #  ,'label1'
        GRU_1 = GRU_1.reset_index()
        if type(GRU_1['datetime'][0]) == str:
            GRU_1['datetime'] = [datetime.strptime(a, '%Y-%m-%d') for a in GRU_1['datetime']]
        
        pred_300 = pd.merge(GRU_1, pred3)
        
        pred_300_1 = pred_300.set_index(['datetime','instrument'])
        pred_300_norm = pred_300_1.groupby(level='datetime', group_keys=False).apply(lambda x:(x-x.mean())/x.std())
        # pred_300_norm.corr(method='spearman')*100
        # pred_300_norm.groupby(level='datetime').apply(lambda x: x['label'+str(300)].corr(x['score1'], method='spearman')).mean()*100

        with open('rankic300.txt','a',encoding='utf-8') as f:
            rankic = pred_300_norm.groupby(level='datetime').apply(lambda x: x['label'+str(300)].corr(x['score1'], method='spearman')).mean()
            f.write(str(rankic*100)+'\t')
            f.write('\n')
            
    # with open('rankic300.txt','a',encoding='utf-8') as f:     
    #     f.write('\n')

