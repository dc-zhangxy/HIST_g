# HIST_gcn

HIST_all/data 中存放instrument数据

HIST_all/HIST_gcn/data 中存放 adj 文件



gcn_models 为 GCN + Individual Information Module

hist_gcn model 为 GCN + Hidden Concept Module + Individual Information Module

hist_gat model 为 GAT + Hidden Concept Module + Individual Information Module



使用特定 adj 只需修改  --adj_name 即可

dataloader_analyst 适配任何与其格式相同的adj文件（dict{date：dataframe} 格式）



主训练/推断文件为 learn_label_adjs.py

 Train = True 时运行 main 训练

 Train = False 时运行 loadandinference 进行推断



    CUDA_VISIBLE_DEVICES=3 python learn_label_adjs.py --model_name GCN --data_set csi300 --adj_name analyst --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/csi300_label11to1_GCN_analyst


```
CUDA_VISIBLE_DEVICES=1 python learn_label_hgat.py --model_name HGAT --data_set all --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/all_label11to1_HGAT
```

