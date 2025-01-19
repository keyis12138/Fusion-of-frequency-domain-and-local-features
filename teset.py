import time
import numpy as np
from torchvision import models
import torch.nn as nn

import patch_generator
import config
import os
import imageio
import torch
from PIL import Image
import torch.fft
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import datareader
from model import trainer


'''dataset = datareader.DataReader()
data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=5)'''
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
img = Image.open('./000_sdv4_00020.png')
imgcv = cv2.imread("./000_sdv4_00020.png")
img_ten = transforms.ToTensor()(img)
# 对每个通道分别进行 FFT
'''fft_channels = []
for c in range(img_ten.shape[0]):  # 遍历 R、G、B 通道
    fft = torch.fft.fft2(img_ten[c])
    fft_shifted = torch.fft.fftshift(fft)  # 中心化频谱
    magnitude = torch.abs(fft_shifted)  # 计算幅值
    log_magnitude = torch.log(1 + magnitude)  # 对数增强
    fft_channels.append(log_magnitude)
processed_tensor = torch.stack(fft_channels, dim=0)  # 形状为 (C, H, W)

# 可视化原图和处理后的图像
processed_image = transforms.ToPILImage()(processed_tensor)
# 可视化每个通道的频谱
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)'''
'''plt.title("Processed Image")
plt.imshow(processed_image)
plt.axis('off')

plt.show()'''
imgcv = cv2.resize(imgcv, (256, 256), cv2.INTER_NEAREST)
img = img.resize((256, 256), Image.LANCZOS)
img = transforms.ToTensor()(img)
print(img.shape)
# img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
grayscale_image1 = cv2.cvtColor(src=imgcv, code=cv2.COLOR_RGB2GRAY)

result, result2 = patch_generator.smash_n_reconstruct(img, True)
print(result.shape)
# result_img = transforms.ToPILImage()(result)

# 显示图片
# result_img.show()
# print(result.shape)
'''result = result/255.0
result_img = transforms.ToPILImage()(result)
result_img.show()'''
# print(result)

# 测试处理后的shape
'''if __name__ == '__main__':
    dataset = datareader.DataReader()
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=3)
    resnet = models.resnet50(pretrained=True)
    model = trainer.CrossAttentionModule(seq_len=config.seq_len,embed_dim=config.embed_dim, num_heads=config.num_heads)
    resnet.fc = nn.Identity()
    fc1 = nn.Linear(2048, 128)
    ac = nn.ReLU()
    tensor_a = torch.rand(32, 2048)  # 输入形状为 (batch_size, seq_len)
    tensor_b = torch.rand(32, 2048)
    embedding1 = ac(fc1(tensor_a))
    embedding1 = embedding1.view(-1,4,32)
    embedding2 = ac(fc1(tensor_b))
    embedding2 = embedding2.view(-1, 4, 32)
    output_tensor = model(embedding1, embedding2)
    #print(output_tensor.shape)
    for i , data in enumerate(dataloader):
        tensorclone = data[0].clone()
        #print(trainer.DFT(data[0]).shape,trainer.select_localfc(tensorclone).shape)
        output1= resnet(trainer.DFT(data[0]))
        output2= resnet(trainer.select_localfc(tensorclone))
        embedding1 = ac(fc1(output1))
        embedding1 = embedding1.view(-1, 4, 32)
        embedding2 = ac(fc1(output2))
        embedding2 = embedding2.view(-1, 4, 32)
        output_tensor = model(embedding1, embedding2)
        lossfc = nn.BCEWithLogitsLoss()
        print(output_tensor.squeeze(1))
        print(data[1])
        loss = lossfc(output_tensor.squeeze(1),data[1])
        print(loss)
        exit(0)'''
