from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
import torch
import torch.nn as nn

 
class ModifiedResNet(nn.Module):
    def __init__(self, resnet):
        super(ModifiedResNet, self).__init__()
        self.resnet = resnet
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.LazyLinear(1)
        
    def forward(self, x, y):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.mean(axis=0)
        x = torch.flatten(x)
        x = torch.cat((x, y.flatten()), 0)
        x = self.fc(x)
        out = self.sigmoid(x)
        return out

def test():
    BATCH_SIZE = 4    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=True).to(device)
    resnet.fc = nn.Linear(512, 1).to(device)
    net = ResNetWithPooling(resnet).to(device)
    input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    print(resnet.fc)
    print(net(input))


if __name__ == "__main__":
    test()
