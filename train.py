import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import random_split

import config
import datareader

sys.path.append('./data')
from earlystop import EarlyStopping
from model.trainer import Trainer

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score


def validate(model, data_loader):
    print("number of validation dataset: ", len(data_loader))
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            input = data[0].cuda()  # [batch_size, 3, height, width]
            label = data[1].cuda()  # [batch_size, 3, 224, 224]
            y_pred.extend(model(input).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)

    return acc


if __name__ == '__main__':
    dataset = datareader.DataReader()
    train_size = int(0.8 * len(dataset))  # 80% 用作训练集
    val_size = len(dataset) - train_size  # 剩余部分作为验证集

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")

    train_writer = SummaryWriter(os.path.join(config.checkpoints_dir, config.name, "train"))
    val_writer = SummaryWriter(os.path.join(config.checkpoints_dir, config.name, "val"))

    model = Trainer()
    early_stopping = EarlyStopping(patience=config.earlystop_epoch, delta=-0.0001, verbose=True)
    for epoch in range(config.epoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            model.total_steps += 1
            epoch_iter += config.batch_size

            model.set_input(data)
            model.optimize_parameters()
            # exit()

            if model.total_steps % config.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % config.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (config.name, epoch, model.total_steps))
                model.save_networks('latest')

        # if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, model.total_steps))
        # model.save_networks('latest')
        model.save_networks(epoch)

        # Validation
        model.eval()
        acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        print("(Val @ epoch {}) acc: {}".format(epoch, acc))
        info = [str(epoch), ',', str(acc)]
        with open('acc_training.txt', 'a') as f:  # path to save the accuracy during training.
            f.writelines(info)
            f.writelines('\n')
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 2, continue training...")
                early_stopping = EarlyStopping(patience=config.earlystop_epoch, delta=-0.00005, verbose=True)
            else:
                print("Learning rate dropped to minimum, still training with minimum learning rate...")
                early_stopping = EarlyStopping(patience=config.earlystop_epoch, delta=-0.00005, verbose=True)
                # break

        model.train()
