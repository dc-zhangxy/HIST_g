import pandas as pd 
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

file_path = 'output/all_transf2424_label1_sche_mse1e-4/rolling2007-01-01/pred.pkl.test0'
with open(file_path,'rb')as f: # 
    pred1=pickle.load(f)
pred1.columns = ['score1','label2']

# file_path2 = 'output/all_transf2424_label2to1_sche_mse1e-4/rolling2007-01-01/pred.pkl.test0'
# with open(file_path2,'rb')as f: # 
#     pred2=pickle.load(f)
# pred2.columns = ['score2','label2']

pred11 =   pred1.reset_index() #  pd.merge(pred1.reset_index(), pred2.reset_index()) # 
pred11['datetime'] = pd.to_datetime(pred11['datetime'] )
df_all = pred11.copy() #pred1.reset_index()  #pd.merge(pred1.reset_index(), pred5.reset_index()) # #  #pred5.reset_index() #
df_all= df_all.set_index(['instrument','datetime'])

scores1 = 'score1'
# (pred11[scores]>0).sum(),(pred11[scores]<0).sum(),(pred11[scores]==0).sum()
df_norm = df_all.copy()
days = 90
df_norm['rolling_mean'] = df_norm.groupby('instrument',as_index=False)[scores1].rolling(window = days,min_periods=1).mean()[scores1]
df_norm['rolling_std'] = df_norm.groupby(['instrument'],as_index=False)[scores1].rolling(window = days,min_periods=1).std()[scores1]
# df_norm = df_norm.fillna(1)
df_norm['normscore'] = (df_norm[scores1]-df_norm['rolling_mean'])/df_norm['rolling_std']
score_for_norm = 'normscore'
rolling_days = 5
df_norm['normscorer'] = df_norm.groupby('instrument',as_index=False)[score_for_norm].rolling(window = rolling_days, min_periods=1).mean()[score_for_norm]

scores = 'normscorer'  #'score1' # 'normscore'  # 
labels = 'label2'
yuzhi = 0.0
df_all = df_norm[df_norm[scores].notna()] #.copy() #pred11.copy()
top_800_stocks = df_all.copy()
low_800_stocks = df_all.copy()
stocks_shaixuan = df_all.copy()
# stocks_shaixuan[scores] = df_all[scores].apply(lambda x: -1 if x ==0 else x)
stocks_shaixuan[scores] = df_all[scores].apply(lambda x: 0 if x < yuzhi and x> -yuzhi  else x)
top_800_stocks[scores] = df_all[scores].apply(lambda x: 0 if x <= yuzhi else x)
low_800_stocks[scores] = df_all[scores].apply(lambda x: 0 if x > -yuzhi else x)
# top_800_stocks['Ref'] = np.sign(top_800_stocks[scores]) * top_800_stocks[labels]
# low_800_stocks['Ref'] = np.sign(low_800_stocks[scores]) * low_800_stocks[labels]
# stocks_shaixuan['Ref'] = np.sign(stocks_shaixuan[scores]) * stocks_shaixuan[labels]
top_800_stocks['Ref'] = (top_800_stocks[scores]) * top_800_stocks[labels]
low_800_stocks['Ref'] = (low_800_stocks[scores]) * low_800_stocks[labels]
stocks_shaixuan['Ref'] = (stocks_shaixuan[scores]) * stocks_shaixuan[labels]

holdcangwei = (top_800_stocks.groupby('datetime')[scores].mean()*df_all.groupby('datetime')[labels].mean()).reset_index()
holdcangwei['TIR'] = holdcangwei[0].cumsum()
holdxiangu = top_800_stocks.copy()
holdxiangu['Ref'] = np.sign(holdxiangu[scores]) * holdxiangu[labels]

# 计算得分的符号乘以标签的值
def cal_cumsum_mancang(top_800_stocks):
    # daily_scores_sum = top_800_stocks.groupby('datetime')['Ref'].mean().reset_index() #sum？
    # daily_scores_sum['TIR'] = daily_scores_sum['Ref'].cumsum()
    daily_scores_sum = (top_800_stocks.groupby('datetime')['Ref'].sum()/top_800_stocks.groupby('datetime')[scores].sum()).reset_index()
    daily_scores_sum['TIR'] = daily_scores_sum[0].cumsum()
    return daily_scores_sum
# 计算得分的符号乘以标签的值
def cal_cumsum_cangwei(top_800_stocks):
    daily_scores_sum = top_800_stocks.groupby('datetime')['Ref'].mean().reset_index() #sum？
    daily_scores_sum['TIR'] = daily_scores_sum['Ref'].cumsum()
    # daily_scores_sum = (top_800_stocks.groupby('datetime')['Ref'].sum()/top_800_stocks.groupby('datetime')[scores].sum()).reset_index()
    # daily_scores_sum['TIR'] = daily_scores_sum[0].cumsum()
    return daily_scores_sum

daily_scores_sum_top_m = cal_cumsum_mancang(top_800_stocks)
daily_scores_sum_low_m = cal_cumsum_mancang(low_800_stocks)
all_score_m = cal_cumsum_mancang(stocks_shaixuan)
hold_xiangu_m = cal_cumsum_mancang(holdxiangu)

daily_scores_sum_top = cal_cumsum_cangwei(top_800_stocks)
daily_scores_sum_low = cal_cumsum_cangwei(low_800_stocks)
all_score = cal_cumsum_cangwei(stocks_shaixuan)
hold_xiangu = cal_cumsum_cangwei(holdxiangu)

# labels = 'label2'
def cal_cumsum_hold(top_800_stocks):
    daily_scores_sum = top_800_stocks.groupby('datetime')[labels].mean().reset_index()
    daily_scores_sum['TIR'] = daily_scores_sum[labels].cumsum()
    return daily_scores_sum
daily_scores_sum_hold = cal_cumsum_hold(df_all)

# # labels = 'label2to1'
# def cal_cumsum_hold_benchmark(top_800_stocks):
#     daily_scores_sum = top_800_stocks.groupby('datetime')['label2to1'].mean().reset_index()
#     daily_scores_sum['TIR'] = daily_scores_sum['label2to1'].cumsum()
#     return daily_scores_sum
# daily_scores_sum_holdhs300 = cal_cumsum_hold_benchmark(df_test_hs300['hs300label'])

js_df = daily_scores_sum_top.dropna().iloc[1:]
bodonglv = js_df['Ref'].std()* np.sqrt(244)
# bodonglv 波动率
nhsyl = ((js_df.iloc[-1]['TIR']+1)-(js_df.iloc[0]['TIR']+1))*(244./len(js_df)) 
# nhsyl 年化收益率

sharpe_r = nhsyl/bodonglv # 1左右 正常
# sharpe ratio 夏普比率
# 依据累乘净值
cum_returns = (1+js_df['Ref']).cumprod()
peak = cum_returns.cummax()
drawdown = (cum_returns- peak)/peak
zuidachuiche_cumprod = drawdown.min()
zdch_time = js_df['datetime'][np.argmin(drawdown)]

#依据累加净值
previos_max =  js_df['TIR'].cummax() # 计算上一个最高点
drawdown_cumsum  = js_df['TIR']-previos_max
zuidachuiche_cumsum = drawdown_cumsum.min()
zdch_time_cumsum = js_df['datetime'][np.argmin(drawdown_cumsum)]

# 双边换手率
# 全满仓
# df_hs = pd.merge(top_800_stocks.reset_index(),top_800_stocks.groupby('datetime')[scores].sum().reset_index().rename(columns={scores: 'scores_sum'}))
# df_hs['cangwei'] = df_hs[scores]/df_hs['scores_sum']
# 按得分算仓位
top_800_stocks['count']=1
df_hs = pd.merge(top_800_stocks.reset_index(),top_800_stocks.groupby('datetime')['count'].sum().reset_index().rename(columns={'count': 'scores_sum'}))
df_hs['cangwei'] = df_hs[scores]/df_hs['scores_sum']
# df_hs = top_800_stocks.copy()
df_hs['prev_score'] = df_hs.groupby('instrument')['cangwei'].shift(1).fillna(0)
# initial_capital = 1
df_hs['buy_amount'] = (df_hs['cangwei'] - df_hs['prev_score']).clip(lower=0)
df_hs['sell_amount'] =  (df_hs['prev_score'] - df_hs['cangwei']).clip(lower=0)
# 计算每日资产价值
# df['daily_return'] = df
huanshou = df_hs.groupby(['datetime']).agg({'buy_amount':'sum', 'sell_amount':'sum' }).sum()*(244/len(js_df)) # 所有股票总换手 # 年化

df_hs['Ref'] = (df_hs['cangwei']) * df_hs[labels]
daily_scores_sum = df_hs.groupby('datetime')['Ref'].sum().reset_index() #sum？
daily_scores_sum['TIR'] = daily_scores_sum['Ref'].cumsum()

df_hs_agg = df_hs.groupby(['datetime']).agg({'Ref':'sum','buy_amount':'sum', 'sell_amount':'sum' })

def cal_net_value_with_fees(df, initial_net_value, transaction_fee_rate):
    df = df.sort_values(by='datetime').copy()
    # df['transaction_fees']
    df['net_value'] = initial_net_value 
    for i in range(1,len(df)):
        previous_net_Values = df.loc[i-1,'net_value']
        daily_return = df.loc[i,'Ref']
        # 计算交易费用
        buy_amount = df.loc[i, 'buy_amount']
        sell_amount = df.loc[i, 'sell_amount']
        transaction_fees = (buy_amount + sell_amount)*transaction_fee_rate

        #
        df.loc[i,'net_value'] = previous_net_Values*(1 + daily_return) - transaction_fees
    return df 

net_value_df = cal_net_value_with_fees(df_hs_agg.reset_index(), 1.0, 0.0005)

qjpjcw = df_hs.groupby('datetime')['cangwei'].sum().mean()
# qjpjcw 区间平均仓位

shenglv = ((df_hs[labels]>0) &(df_hs[scores]>0) ).sum()/(df_hs[scores]>0).sum()
# shenglv 胜率

peilv_a = df_hs[(df_hs[scores]>0) & (df_hs[labels]>0)][labels].sum()
peilv_b = df_hs[(df_hs[scores]>0) & (df_hs[labels]<0)][labels].sum()
peilv = - peilv_a/peilv_b
# peilv 赔率

kzzb = (df_hs[scores]>0).sum()/len(df_hs)
kdzb = (df_hs[scores]<0).sum()/len(df_hs)
# kzzb, kdzb 看涨占比

# # 超额收益
chaoe = pd.merge(daily_scores_sum, daily_scores_sum_hold, on=['datetime']).iloc[2:]
chaoe['chaoe'] = chaoe['TIR_x'] - chaoe['TIR_y']

with open('acf0.txt','a',encoding='utf-8') as f:
    # for i in [0.0,0.0002,0.001,0.003,]:
        for j in [bodonglv, nhsyl, sharpe_r, zuidachuiche_cumsum, zdch_time_cumsum, \
            huanshou['buy_amount'], huanshou['sell_amount'], qjpjcw, shenglv, peilv, kzzb, ]:
            f.write(str(j)+'\t')
        f.write('\n')
    # f.write('\n')



signscore = top_800_stocks.groupby('datetime')[scores].sum().reset_index()

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.plot(daily_scores_sum_holdhs300['datetime'],daily_scores_sum_holdhs300['TIR'],label='hs300',color ='#1f77b4')
# ax1.plot(net_value_df['datetime'],net_value_df['net_value']-1,label='koufeihou',color ='black')
# ax1.plot(net_value_df['datetime'],(1+net_value_df['Ref']).cumprod()-1,label='koufeiqian',color ='gray')
# plt.figure(figsize=(6,12))
ax2.plot(signscore['datetime'],signscore[scores],alpha=0.5,color='gray',label='signscore')
ax1.plot(daily_scores_sum_hold['datetime'],daily_scores_sum_hold['TIR'],label='hold',color ='#1f77b4')
ax1.plot(daily_scores_sum_hold['datetime'],0.5*daily_scores_sum_hold['TIR'],label='half hold',color ='#ff7f0e')
# plt.plot(daily_scores_sum_low['datetime'],daily_scores_sum_low['TIR'],label='short',color ='#2ca02c')
# plt.plot(all_score['datetime'],all_score['TIR'],label='long+short',color ='#17becf')
# ax1.plot(daily_scores_sum_top_m['datetime'],daily_scores_sum_top_m['TIR'],label='long_mancang',color ='#2ca02c')
# plt.plot(hold_xiangu_m['datetime'],hold_xiangu_m['TIR'],label='mancang_xuangu',color ='#17becf')
ax1.plot(holdcangwei['datetime'],holdcangwei['TIR'],label='hold position',color ='green')
# plt.plot(hold_xiangu['datetime'],hold_xiangu['TIR'],label = 'hold_xiangu',color = 'pink')
ax1.plot(daily_scores_sum_top['datetime'],daily_scores_sum_top['TIR'],label='long',color ='#d62728')
# ax1.plot(daily_scores_sum['datetime'],daily_scores_sum['TIR'],label='long2',color ='orange')
ax1.plot(daily_scores_sum_top['datetime'],daily_scores_sum_top['TIR']-holdcangwei['TIR'],label='long-chold',color ='#17becf')
# plt.plot(scoreand300['datetime'],scoreand300['TIR_x']- scoreand300['TIR_y'],label='chaoe',color ='pink')

ax1.legend(fontsize = 10,bbox_to_anchor=(1.1,0), loc=3, borderaxespad=0 )
ax1.xaxis.set_tick_params(rotation=15)
ax1.set_ylabel('TIR')
plt.title('stock label1')
# mean, 标准化
# fig.subplots_adjust(right=1.2)
fig.savefig('stock label1——test.png',bbox_inches='tight')


