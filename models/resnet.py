from torch import nn


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 3):
        super(ResNet, self).__init__()
        self.infilters = 64
        self.conv1 = nn.Sequential(
                        nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU())
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool3d(4, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or self.infilters != filters:

            downsample = nn.Sequential(
                nn.Conv3d(self.infilters, filters, kernel_size=1, stride=stride),
                nn.BatchNorm3d(filters),
            )
        layers = []
        layers.append(block(self.infilters, filters, stride, downsample))
        self.infilters = filters
        for i in range(1, blocks):
            layers.append(block(self.infilters, filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    pass