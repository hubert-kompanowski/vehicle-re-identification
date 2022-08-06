import torch
from torchvision.models import ResNet50_Weights, resnet50

net = resnet50(weights=ResNet50_Weights.DEFAULT)


x = torch.randn((1,3,256,256))

# print(net(x).shape)/

count = 0
for x in net.children():
    if count <= 5:
        for p in x.parameters():
            p.requires_grad = False
    print(count, x)

    count +=1


# print(net)