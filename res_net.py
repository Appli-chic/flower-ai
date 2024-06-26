from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Linear, LogSoftmax


class ResNet(Module):
    def __init__(self, classes=5):
        super(ResNet, self).__init__()
        layers = [3, 3, 3, 3]
        block = ResidualBlock
        self.inplanes = 16
        self.conv1 = Sequential(
            Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(16),
            ReLU()
        )
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = AvgPool2d(6, stride=1)
        self.fc = Linear(128, classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential(
                Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, input_tensor):
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.maxpool(input_tensor)
        input_tensor = self.layer0(input_tensor)
        input_tensor = self.layer1(input_tensor)
        input_tensor = self.layer2(input_tensor)
        input_tensor = self.layer3(input_tensor)

        input_tensor = self.avgpool(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc(input_tensor)

        return self.logSoftmax(input_tensor)


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU())
        self.conv2 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
