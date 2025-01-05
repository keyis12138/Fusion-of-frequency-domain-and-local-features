import config
import os
from PIL import Image
import torchvision.transforms as transforms
class DataReader():
    def __init__(self):
        self.classes = config.cls
        self.root = config.dataroot
        self.img = []
        self.label = []
        for cls in self.classes:
            real_img_list = [os.path.join(self.root + '/' + cls, '0_real', train_file) for train_file in
                             os.listdir(os.path.join(self.root + '/' + cls, '0_real'))]

            real_label_list = [0 for _ in range(len(real_img_list))]

            fake_img_list = [os.path.join(self.root + '/' + cls, '1_fake', train_file) for train_file in
                             os.listdir(os.path.join(self.root + '/' + cls, '1_fake'))]

            fake_label_list = [1 for _ in range(len(fake_img_list))]

            self.img = self.img + real_img_list + fake_img_list
            self.label = self.label + real_label_list + fake_label_list

        print(f'directory:{self.root}, realimg:{len(real_img_list)}, fakeimg:{len(fake_img_list)}')

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img, target =Image.open(self.img[index]).convert('RGB'), self.label[index]
        # img = img.resize((256, 256), Image.ANTIALIAS)旧版pillow
        img = img.resize((256, 256), Image.LANCZOS)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = self.transform(img)
        return img, target