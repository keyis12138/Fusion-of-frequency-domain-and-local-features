dataroot = './dataset/'#训练集根目录
cls = ['airplane']#加载的训练集类别
#cls = ['airplane','bird','bicycle','boat','bottle','bus','car','cat','cow','chair','diningtable','dog','person','pottedplant','motorbike','tvmonitor','train','sheep','sofa','horse']
optim = 'adam'#优化器
beta1 = 0.9
#optim = 'sgd'
lr = 0.00005#学习率
batch_size =32
gpu_ids = 0