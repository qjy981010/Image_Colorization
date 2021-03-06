import torch
import torch.nn as nn
import math


class LowLevelNet(nn.Module):
    """
    """

    def __init__(self, ):
        super(LowLevelNet, self).__init__()
        self.cnn = self._get_cnn_layers()
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        return x

    def _get_cnn_layers(self):
        channels = (1, 64, 128, 128, 256, 256, 512)
        cnn_layers = []
        for i in range(1, len(channels)):
            cnn_layers.append(nn.Conv2d(channels[i-1], channels[i], 3, i%2+1, 1))
            cnn_layers.append(nn.BatchNorm2d(channels[i]))
            cnn_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class GlobalLevelNet(nn.Module):
    """
    """
    def __init__(self):
        super(GlobalLevelNet, self).__init__()
        self.cnn = self._get_cnn_layers()
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.global_out_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        classify_input = self.fc(x)
        global_feature = self.global_out_layer(classify_input)
        return classify_input, global_feature

    def _get_cnn_layers(self):
        cnn_layers = []
        for i in range(4):
            cnn_layers.append(nn.Conv2d(512, 512, 3, 2-i%2, 1))
            cnn_layers.append(nn.BatchNorm2d(512))
            cnn_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ClassificationNet(nn.Module):
    """
    """
    def __init__(self, class_num):
        super(ClassificationNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MidLevelNet(nn.Module):
    """
    """
    def __init__(self):
        super(MidLevelNet, self).__init__()
        self.cnn = self._get_cnn_layers()
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        return x

    def _get_cnn_layers(self):
        cnn_layers = []
        channels = (512, 256)
        for channel in channels:
            cnn_layers.append(nn.Conv2d(512, channel, 3, 1, 1))
            cnn_layers.append(nn.BatchNorm2d(channel))
            cnn_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FusionNet(nn.Module):
    """
    """
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, mid_result, global_result):
        height, width = mid_result.size()[2:]
        global_result = global_result.unsqueeze(2).unsqueeze(3)
        global_result = global_result.repeat(1, 1, height, width)
        result = torch.cat((mid_result, global_result), 1)

        result = self.conv(result)
        result = self.relu(self.bn(result))
        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ColorizationPart(nn.Module):
    """
    """
    def __init__(self):
        super(ColorizationPart, self).__init__()
        self.cnn = self._get_cnn_layers()
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        return x

    def _get_cnn_layers(self):
        cnn_layers = []
        in_channel = 256
        channels = (128, 'U', 64, 64, 'U', 32, 2, 'U')
        for channel in channels[1:]:
            if channel == 'U':
                cnn_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            else:
                cnn_layers.append(nn.Conv2d(in_channel, channel, 3, 1, 1))
                cnn_layers.append(nn.BatchNorm2d(channel))
                if channel == 2:
                    cnn_layers.append(nn.Sigmoid())
                else:
                    cnn_layers.append(nn.ReLU(inplace=True))
                in_channel = channel
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ColorizationNet(nn.Module):
    """
    """
    def __init__(self, class_num):
        super(ColorizationNet, self).__init__()
        self.low_level_net = LowLevelNet()
        self.global_level_net = GlobalLevelNet()
        self.classifier = ClassificationNet(class_num)
        self.mid_level_net = MidLevelNet()
        self.fusion = FusionNet()
        self.colorization = ColorizationPart()

    def forward(self, x):
        x = self.low_level_net(x)
        classify_input, global_result = self.global_level_net(x)
        classify_result = self.classifier(classify_input)
        mid_result = self.mid_level_net(x)
        fusion_result = self.fusion(mid_result, global_result)
        result = self.colorization(fusion_result)
        return classify_result, result
