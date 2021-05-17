'''
    准备训练所需的数据，使用pytorch自带的loader
'''

# --------------------------------准备训练数据--------------------------------
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, mode, txt, transform=None, target_transform=None, loader=default_loader):
        self.mode = mode
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')     # 移除字符串首尾的换行符
            line = line.rstrip()        # 删除末尾空
            words = line.split()        # 以空格为分隔符 将字符串分成

            if mode == 'train':
                imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
            else:
                imgs.append((words[0]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        if self.mode == 'train':
            fn, label = self.imgs[index]
        else:
            fn = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == 'train':
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.imgs)

trans = transforms.Compose([
                            transforms.Resize(size=(224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.6020298, 0.5084915, 0.38683322], std=[0.2768252, 0.29009122, 0.3120243]),  #归一化
                            # 以下为一些数据增强方式
                            transforms.RandomHorizontalFlip(p=0.4),
                            transforms.CenterCrop(size=224),
                            transforms.RandomVerticalFlip(p=0.4),
                            transforms.RandomRotation(degrees=15)
                             ])
test_trans = transforms.Compose([transforms.Resize(size=(224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.6020298, 0.5084915, 0.38683322], std=[0.2768252, 0.29009122, 0.3120243])
                                ])

# 加载数据并做预处理
train_data = MyDataset(mode='train', txt='train.txt', transform=trans)
val_data = MyDataset(mode='train', txt='val.txt', transform=trans)
test_data = MyDataset(mode='test', txt='test.txt', transform=test_trans)

batch_size = 16

# 调用DataLoader批量加载数据
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

train_iter = int(len(train_data) / batch_size)
test_iter = int(len(test_data) / batch_size)