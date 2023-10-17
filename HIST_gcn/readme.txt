CUDA_VISIBLE_DEVICES=2 python learn_label.py --model_name GCN --data_set all_industry --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/all_ind2_label11to1_HGCN

CUDA_VISIBLE_DEVICES=3 python learn_label_adjs.py --model_name GCN --data_set csi300 --hidden_size 128 --num_layers 2 --labels 11 --repeat 1 --outdir ./output/csi300_label11to1_GCN_analyst
  