import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.tsa.api as smt

labels = pickle.load (open('HIST_newdata/output51/label1to21.pkl','rb') ).reset_index()

# dataname = 'all_'+'HIST'+''
modelname = ['GRU','HIST','HISTdelpre']
labelname = ['','_label2to1','_label6to1','_label11to1']
# datanames = ['all_ind1' + '_label11to1_' + 'HGCN',
#             'all_ind2' + '_label11to1_' + 'HGCN']
# datanames = ['all_label11to1_HGCN1_amend_analyst',
#             'all_label11to1_HGCN1_amend_prod',
#             'all_flow_label11to1_HGCN1_flow']
# labels['instrument'] =  labels['instrument'].apply(lambda x: str(x)[2:])

# datanames = ['all_label21to1_GRU',
#             'all_label21to1_HISTdelpre',
#             'allind1_label21_GCNind1_0930',
#             'all_label21to1_GRU_icloss',
#             'all_label21to1_HISTdelpre_icloss',
#             'allind1_label21_GCN_icloss_1130',]
# datanames = ['all_label11to1_HGAT_09']
datanames = ['comb',
            'comb21',
            'combGHC21',
            'combHC21']

# for b in labelname:
#     for a in modelname:
for d in datanames:
        # dataname = 'all_'+a+b
        # dir_ = 'HIST_newdata/output51/'+dataname+'/rolling2009-01-01/pred.pkl.test0'
        dir_ = 'HIST_gcn/output51gcn/'+d+'/rolling2009-01-01/pred.pkl.test0'
        # dir_ = 'HIST_gcn/output53gcn/'+d+'/rolling2005-01-01/pred.pkl.test0'

        GRU_1=pickle.load(open(dir_,'rb'))
        GRU_1.columns = ['score1','label1']
        GRU_1 = GRU_1.reset_index()
        if type(GRU_1['datetime'][0]) == str:
            GRU_1['datetime'] = [datetime.strptime(a, '%Y-%m-%d') for a in GRU_1['datetime']]

        pred_all = pd.merge(GRU_1.drop('label1',axis=1), labels)
        pred_all1 = pred_all.set_index(['datetime','instrument'])
        pred_all_norm = pred_all1.groupby(level='datetime', group_keys=False).apply(lambda x:(x-x.mean())/x.std())
        # pred_all_norm.corr(method='spearman')*100
        # with open('acf.txt','a',encoding='utf-8') as f:
        #     for i in [1,2,6,11,21]:
        #         rankic = pred_all_norm.groupby(level='datetime').apply(lambda x: x['label'+str(i)].corr(x['score1'], method='spearman')).mean()
        #         f.write(str(rankic*100)+'\t')
        #     f.write('\n')
        acf1 = []
        acf5 = []

        stock300 = pred_all_norm.reset_index()['instrument'].drop_duplicates()
        lens = len(stock300)
        pred_cal = pred_all_norm.reset_index().set_index(['instrument','datetime'])

        for a in stock300:
            cal_ser = pred_cal.loc[a]
            if len(cal_ser)>10:
                acf = smt.stattools.acf(cal_ser['score1'])
                acf1.append(acf[1])
                acf5.append(acf[5])
            # else:
            #     lens = lens-1
        # print(np.mean(acf1),np.mean(acf5))
        with open('acf.txt','a',encoding='utf-8') as f:
            f.write(str(np.mean(acf1))+'\t')
            f.write(str(np.mean(acf5))+'\t')
            f.write('\n')
    # with open('acf.txt','a',encoding='utf-8') as f:
    #     f.write('\n')

