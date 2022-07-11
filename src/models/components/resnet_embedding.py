from torchvision.models import ResNet50_Weights, resnet50

net = resnet50(weights=ResNet50_Weights.DEFAULT)


print(net)
