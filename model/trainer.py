import torch
import torch.nn as nn
import torchvision.models as models
from model.base_model import BaseModel
import config
'''
dataloader
model

save
(可选）
options
'''


class xxmodel(nn.model):
    def __init__(self):
        super(xxmodel,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(2048,128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128,1)
    def forward(self,x):
        output = 0
        return output


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
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
        if len(opt.gpu_ids) == 0:
            self.model.to('cpu')
        else:
            self.model.to(config.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input = data[0].to(self.device)
        self.label = data[1].to(self.device)
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