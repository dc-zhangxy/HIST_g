# HIST_gcn

HIST_gcn/data 中存放 adj 文件

gcn_models 为 GCN + Individual Information Module

hist_gcn models 为 GCN + Hidden Concept Module + Individual Information Module



dataloader_analyst 为 analyst 和 product adj

dataloader_flow 为 inflow_trade adj

dataloader_industry 为 industry adj 



使用特定 adj 需在 learn_label_adjs.py 中修改路径



CUDA_VISIBLE_DEVICES=2 python learn_label.py --model_name GCN --data_set all_industry --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/all_ind2_label11to1_HGCN

CUDA_VISIBLE_DEVICES=3 python learn_label_adjs.py --model_name GCN --data_set csi300 --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/csi300_label11to1_GCN_analyst
