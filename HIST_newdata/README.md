# HIST
HIST依靠qlib完成数据加载和处理，因此需要先配置qlib环境并下载数据


## 运行环境：
可参考以下配置
GPU: RTX 3090 (24GB) 
CPU: 24 vCPU AMD EPYC 7642 48-Core Processor
内存: 80GB (在运行全A数据时内存非常重要，不能过低)

PyTorch  1.11.0
Python  3.8(ubuntu20.04)
Cuda  11.3

## qlib安装与数据下载

### qlib数据下载
```
# install Qlib from source
pip install numpy
pip install --upgrade  cython
pip install pyqlib
git clone https://github.com/microsoft/qlib.git && cd qlib
pip install .

# Download the stock features of Alpha360 from Qlib
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

cd ~/HIST_all
```
* qlib会频繁更新，下载命令会随之变化。如以上流程出现问题可参照https://github.com/microsoft/qlib的方式下载cn 1d数据。

### 补充数据下载
```
mkdir data/cn_data_updated
wget https://github.com/chenditc/investment_data/releases/download/2023-06-20/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ./data/cn_data_updated --strip-components=2
```
## 其他环境配置

```
pip install catboost
pip install xgboost
pip install pickle5
 ```
如果运行时出现 no module named xxx，意味需要安装相应的库。
需要安装pytorch和cuda，请根据显卡版本匹配安装。
### HIST原文复现
先进入HIST文件夹，如果此时在qlib文件夹下：
```
cd ./HIST
```
CSI100和CSI300复现，output_example中放有此前跑出的原文复现模型样例可供参考。
```
# CSI 100
python learn.py --model_name HIST --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_HIST

# CSI 300
python learn.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_HIST
```
output文件夹中，info.json是包含输入信息及结果的总结文件，pred.pkl.test/train/valid{time}是每次训练的最佳模型在inference的时候产生的结果，如需打开
```
# 以第一次训练的测试结果为例
import pickle
with open('{output_path}/pred.pkl.test0','rb') as file:
    test = pickle.load(file)
```
* 原文为了测试效果稳定性会重复训练10次模型计算标准差，如果需要改动可以在上述命令中加入--repeat，例如复现CSI100时如果只想要训练一次：
```
# CSI 100
python learn.py --model_name HIST --data_set csi100 --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/csi100_HIST
```

### 复现阶段其他模型

为了节省时间，此处均使用了一次训练
* GRU
```
# GRU on CSI 100
python learn.py --model_name GRU --data_set csi100 --hidden_size 128 --num_layers 2 --repeat 1 --outdir ./output/csi100_GRU

# GRU on CSI 300
python learn.py --model_name GRU --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_GRU --repeat 1
```
* 不同标签测试
```
# 均以100为例

#单日预测
python learn_label.py --model_name HIST --labels 2 --data_set csi100 --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/csi100_HIST_label2to1

#五日预测
python learn_label.py --model_name HIST --labels 6 --data_set csi100 --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/csi100_HIST_label6to1

#十日预测
python learn_label.py --model_name HIST --labels 11 --data_set csi100 --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/csi100_HIST_label11to1
```
* 无预定义测试
```
python learn_noconcept.py --model_name HIST --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_no_concept --repeat 1

python learn_noconcept.py --model_name HIST --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_no_concept --repeat 1
```
* 滚动测试
```
python learn_roll.py --model_name HIST --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_rolling --repeat 1

python learn_roll.py --model_name HIST --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_rolling --repeat 1
```

### 拓展
```
# 退出HIST，来到HIST_newdata
cd ../HIST_newdata
```
在更新的数据中，提供了300，500和全A的股票，由于500和全A体量较大，固定训练周期数据量很大对服务器要求高，因此500和全A都只进行了滚动训练。如服务器可以承载500和全A的数据量，可将--data_set 更换为csi500 和all，并注意修改outdir

所有无预定义的结果仅需要进入learn系列的py文件，将30行的model改为model2，请注意同时修改outdir以免混淆结果。

```
# CSI 300
python learn.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_HIST
```
* 滚动训练
```
sh 300_roll.sh
sh 500_roll.sh
sh all_roll.sh
```
全A五日
```
sh all_roll_5day.sh
```
### 收益部分
在计算收益时，首先需将test部分的结果转换为csv文件
```
import pickle
import pandas as pd

with open('{output_path}/pred.pkl.test0','rb') as file:
    test = pickle.load(file)
test.to_csv('{name}.csv')
```
具体说明请参照calreturn.py，需要根据具体使用情况更改代码及匹配数据。代码中使用了tushare pro的数据，可加入token或替换为相似数据集。
