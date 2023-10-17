import os

# dataname = 'steel' #  'MSRCv2'  #'lost' 

dataset = 'all' 
model = 'HISTdelpre'
# cuda = 1

# parx = 0.1 #0.5
# parf = 1 # 1 #1
# parp = 0.1 # 0.05 #0.1 #0.05

# kfoldt = 1

# datalist = ['lost']  # BirdSong SoccerPlayer lost

# for dataname in datalist:
#     for kfoldt in range(5): # 5
        # os.system('CUDA_VISIBLE_DEVICES=2 python multi_trainer_weight.py --dataname %s --datadir %s --kfold %i --cuda %i --parx %f --parf %f --parp %f' % (dataname, datadir, kfoldt, cuda, parx, parf, parp))
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name HIST --labels 2 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_HISTdelpre_label2to1' )
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name HIST --labels 6 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_HISTdelpre_label6to1' )
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name HIST --labels 11 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_HISTdelpre_label11to1' )
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name GRU --labels 2 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_GRU_label2to1' )
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name GRU --labels 6 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_GRU_label6to1' )
# os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name GRU --labels 11 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_GRU_label11to1' )
        
os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name HIST --labels 6 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_HIST_label6to1' )
os.system('CUDA_VISIBLE_DEVICES=1 python learn_label.py --model_name HIST --labels 11 --data_set all --hidden_size 128 --repeat 1 --num_layers 2 --outdir ./output/all_HIST_label11to1' )
