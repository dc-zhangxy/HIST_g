import torch
import numpy as np
import pandas as pd
import feather
from sklearn.preprocessing import StandardScaler
from operator import itemgetter


class DataLoader:

    def __init__(self, df_feature, df_label, np_adj, ind_stock, time_index, device=None, pin_memory=False):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature #.values
        self.df_label = df_label # .values
        self.np_adj = np_adj
        self.ind_stock = ind_stock
        self.ind_stock_dict = dict(zip(ind_stock,range(len(ind_stock))))
        self.device = device
        self.time_index = time_index

        # if pin_memory:
        #     self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
        #     self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
        #     self.df_adj = torch.tensor(self.df_adj, dtype=torch.float, device=device)
        #     self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.pin_memory = pin_memory

    def get(self, slc):  # 之后需打乱batch顺序
        day = self.time_index[slc]
        
        stock_today = self.df_label.loc[day].index
        stock_today_dict = dict(zip(stock_today,range(len(stock_today))))
        
        stock_inner = list(set(stock_today) & set(self.ind_stock))
        
        # stock_index = itemgetter(*stock_today)(self.ind_stock_dict)
        stock_ind_index = itemgetter(*stock_inner)(self.ind_stock_dict)
        stock_today_index = itemgetter(*stock_inner)(stock_today_dict)
        
        adj_today = pd.DataFrame(np.zeros([len(stock_today), len(stock_today)])) # np.identity(len(stock_today)) # 
        adj_today.loc[stock_today_index, stock_today_index] = np.array(self.np_adj)[stock_ind_index,:][:,stock_ind_index]
        # adj_today = np.array(adj_today) + np.identity(len(stock_today)) # 令对角线为1
        adj_today = np.array(adj_today)
        # np.fill_diagonal(B, 0)
        adj_today[np.diag_indices(adj_today.shape[0])] = 1.
        
        # adj_out = torch.tensor(self.np_adj[stock_index,:][:,stock_index], device=self.device)
        adj_out= torch.tensor(adj_today, dtype=torch.float, device=self.device)
        # print(adj_out.dtype) # torch.float64
        sumW = torch.einsum('ij->i', adj_out)
        sumW[sumW==0] = 1.
        sumW = torch.diag(1/sumW)
        H = torch.mm(adj_out, sumW)
        
        outs = self.df_feature.loc[day].values, self.df_label.loc[day].values[:,0],
        # outs = self.df_feature[slc], self.df_label[slc] ind = [1,2,3]

        # if not self.pin_memory:  # 每次放device还是一次性放的区别
        outs = tuple(torch.tensor(x, dtype=torch.float, device=self.device) for x in outs)

        return outs + (H, stock_today, day,)  # (self.index[slc],)
    


if __name__ == '__main__':

    data_set='csi800'
    data_dir = 'data/csi800/'

    industry = pd.read_excel(data_dir+'industry.xlsm')  # 行业数据
    industry800 = industry[industry['是否CSI800']=='是'].reset_index(drop=True)
    stock800list = [a[:-3] for a in industry800['stock_code']]
    stock_dict = dict(zip(stock800list,range(len(stock800list))))

    np_adj = np.zeros([800,800])
    for i in range(800):
        for j in range(800):
            if i!=j and industry800['行业1'][i] == industry800['行业1'][j]:
                np_adj[i,j] = 1

    df_train_new = feather.read_dataframe(data_dir+"hqdf800_train_label1.feather")
    df_mul = df_train_new.set_index(['END_DATE','STOCK_CODE'], drop=True)
    
    total_time = df_train_new['END_DATE'].drop_duplicates().to_list()
    time_index = [str(t)[:10] for t in total_time]
    total_time_len = len(time_index)
    
    train_end_index = int(total_time_len*0.8)
    valid_end_index = int(total_time_len*0.9)
    train_start_date = time_index[0] # '2013-02-21'
    train_end_date = time_index[train_end_index] #'2017-12-31'
    valid_start_date = time_index[train_end_index+1] #'2018-01-01'
    valid_end_date = time_index[valid_end_index] # '2020-12-31'
    test_start_date = time_index[valid_end_index+1] #'2021-01-01'
    test_end_date = time_index[-1] # '2022-12-29'

    # df_label = df_mul['label']
    # df_feature = df_mul.drop('label', axis=1)

    transfer = StandardScaler()
    array_label = transfer.fit_transform(np.array(df_mul['label1']).reshape(-1, 1))
    array_feature = transfer.fit_transform(np.array(df_mul.drop('label1', axis=1).reset_index(drop=True)))

    df_features = pd.DataFrame(array_feature, index = df_mul.index, columns=df_mul.columns[1:])
    df_label = pd.DataFrame(array_label, index = df_mul.index, columns=['label1'])

    df_train = df_features.loc[time_index[:train_end_index]]
    df_valid = df_features.loc[time_index[train_end_index+1:valid_end_index]]
    df_test = df_features.loc[time_index[valid_end_index+1:]]
    # df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], 
    #                             col_set=["feature", "label"]) # split
    label_train = df_label.loc[time_index[:train_end_index]]
    label_valid = df_label.loc[time_index[train_end_index+1:valid_end_index]]
    label_test = df_label.loc[time_index[valid_end_index+1:]]
    
    train_loader = DataLoader(df_train, label_train, np_adj, 
                stock_dict, device = 'cuda')
    
    for slc in range(train_end_index):
        # np.random.shuffle(indices)
        # global_step += 1
        feature, label, adj , stock_index, day_index = train_loader.get(slc)

        print(feature.shape, label.shape, adj.shape , stock_index.shape, day_index)
        # break
        # if args.model_name == 'HIST':
        #     pred = model(feature, adj)
