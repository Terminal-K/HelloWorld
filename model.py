from torchvision import models
import torch.nn as nn

# -----------------------------------------定义Resnet50结构-----------------------------------------
'''
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=(7,7),stride=stride,padding=(3,3), bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
#
class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(1,1),stride=(1,1), bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
#
class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=172, expansion = 4):
        super(real_ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=256, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=2)
        self.fc = nn.Linear(1024,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.layer1(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.layer2(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.layer3(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.layer4(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))

        x = self.avgpool(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = x.view(batch_size,-1)
        # print(x.size(0), x.size(1))
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])
'''

class Res50Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=172, drop_rate=0.4):
        super(Res50Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet50(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7

    def forward(self, x):
        x = self.features(x)
        x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

#--------------------------------实例化模型--------------------------------
# model = ResNet50().cuda()
# 预训练模型↓
model = Res50Feature().cuda()

print(model)        #查看模型结构

import torch
#--------------------------------配置损失函数与优化器--------------------------------
LossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.999)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1204)   # 似乎余弦退火的学习率衰减方法对本任务效果并不好