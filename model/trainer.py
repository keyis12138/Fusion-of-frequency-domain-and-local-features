import torch
import torch.nn as nn
import torchvision.models as models
from model.base_model import BaseModel
import config
import patch_generator
import datareader
'''
dataloader
model

save
(可选）
options
'''


def DFT(input):
    if input.dim() != 4 or input.shape[1] != 3:
        raise ValueError("输入张量必须是形状为 (batchsize, 3, H, W) 的四维张量.")
    processed_batch = []
    for tensor in input:  # 遍历每个样本 (3, H, W)
        processed_channels = []
        for c in range(tensor.shape[0]):  # 遍历 R、G、B 通道
            fft = torch.fft.fft2(tensor[c])
            fft_shifted = torch.fft.fftshift(fft)  # 中心化频谱
            magnitude = torch.abs(fft_shifted)  # 计算幅值
            log_magnitude = torch.log(1 + magnitude)  # 对数增强
            processed_channels.append(log_magnitude)
        processed_tensor = torch.stack(processed_channels, dim=0)  # 形状为 (C, H, W)
        processed_batch.append(processed_tensor)
    return torch.stack(processed_batch, dim=0)  # (batchsize, 3, H, W)


def select_localfc(input):
    processed_batch = []
    for tensor in input:
        result, _ = patch_generator.smash_n_reconstruct(tensor)
        processed_batch.append(result)
    return torch.stack(processed_batch, dim=0)  # (batchsize, 3, H, W)


class CrossAttentionModule(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads):
        """
        初始化交叉注意力模块。

        参数:
            embed_dim (int): 特征维度（每个向量的长度）。
            num_heads (int): 多头注意力的头数。
        """
        super(CrossAttentionModule, self).__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim * seq_len, 512)  # 将8*32的特征展平到512维
        self.fc2 = nn.Linear(512, 256)  # 隐藏层
        self.fc3 = nn.Linear(256, 1)  # 输出层：二分类的输出是一个值

        # 激活函数
        self.relu = nn.ReLU()

        # Sigmoid 激活函数用于输出层进行二分类概率预测
        self.sigmoid = nn.Sigmoid()

        self.cross_attention_a_to_b = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention_b_to_a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tensor_a, tensor_b):
        """
        执行交叉注意力计算。

        参数:
            tensor_a (torch.Tensor): 输入张量 A，形状为 (batch_size, seq_len_a, embed_dim)。
            tensor_b (torch.Tensor): 输入张量 B，形状为 (batch_size, seq_len_b, embed_dim)。

        返回:
            torch.Tensor: 拼接后的特征张量，形状为 (batch_size, seq_len_a + seq_len_b, embed_dim)。
        """
        # A -> B 的注意力
        enhanced_a, _ = self.cross_attention_a_to_b(tensor_a, tensor_b, tensor_b)  # Q=A, K=B, V=B
        # B -> A 的注意力
        enhanced_b, _ = self.cross_attention_b_to_a(tensor_b, tensor_a, tensor_a)  # Q=B, K=A, V=A

        # 拼接增强后的特征
        concatenated = torch.cat([enhanced_a, enhanced_b], dim=1)  # 在序列长度维度拼接
        x = concatenated.flatten(start_dim=1)  # 形状变为 (batch_size, seq_len * embed_dim)

        # 前向传播
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + ReLU 激活
        x = self.fc3(x)  # 输出层，输出一个值

        # 对于二分类，使用 sigmoid 转换为概率
        # x = self.sigmoid(x)
        return x


class xxmodel(nn.Module):
    def __init__(self):
        super(xxmodel,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.model = CrossAttentionModule(seq_len=config.seq_len, embed_dim=config.embed_dim, num_heads=config.num_heads)
        self.fc1 = nn.Linear(2048,128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128,1)
    def forward(self,x):
        xlocal = x.clone()
        x = self.resnet(DFT(x))
        xlocal = select_localfc(xlocal)
        xlocal = self.resnet(xlocal)
        x = self.ac(self.fc1(x))
        x = x.view(-1, 4, 32)
        xlocal = self.ac(self.fc1(xlocal))
        xlocal = xlocal.view(-1, 4, 32)
        output = self.model(x, xlocal)
        return output


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self):
        super(Trainer, self).__init__()
        self.model = xxmodel()
        '''if self.isTrain and not opt.continue_train:
            self.model = Patch5Model()
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if not self.isTrain or opt.continue_train:
            # self.model = resnet50(num_classes=1)
            self.model = Patch5Model()
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)'''
        self.loss_fn = nn.BCEWithLogitsLoss()
        if config.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config.lr, betas=(config.beta1, 0.999))
        elif config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=config.lr, momentum=0.0, weight_decay=0)
        self.model.to(torch.device("cuda"))

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input = data[0].to(self.device)
        self.label = data[1].to(self.device).float()
    def forward(self):
        self.output = self.model(self.input)
    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)
    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()