import torch

dataroot = './dataset/'#训练集根目录
cls = ['airplane']#加载的训练集类别
#cls = ['airplane','bird','bicycle','boat','bottle','bus','car','cat','cow','chair','diningtable','dog','person','pottedplant','motorbike','tvmonitor','train','sheep','sofa','horse']
optim = 'adam'#优化器
beta1 = 0.9
#optim = 'sgd'
lr = 0.00005#学习率
batch_size = 64
gpu_ids = 1
embed_dim = 32  # 特征维度
num_heads = 4  # 多头数量
seq_len = 8  # resnet输出特征长度
earlystop_epoch = 5
checkpoints_dir = './checkpoints'
device = torch.device("cuda" if torch.cuda.is_available() else "")
epoch = 2
loss_freq = 200
save_latest_freq = 2000
name = 'experiment_name'
isTrain = True
num_workers = 5
